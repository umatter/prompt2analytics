//! Egress policy for the LLM provider HTTP client.
//!
//! The LLM routes make server-side requests to a provider endpoint the
//! caller may choose (`base_url`). This module decides which endpoints the
//! server connects to:
//!
//! - [`EgressPolicy::check_base_url`] vets the URL itself: an optional host
//!   allowlist, the scheme (`https`; `http` only for a loopback Ollama
//!   endpoint) and IP-literal hosts through [`is_disallowed_ip`].
//! - [`GuardedResolver`] is the client's DNS resolver. It resolves a name,
//!   refuses the whole name when any answer is an internal address, and hands
//!   reqwest only the addresses it checked, so the check and the connection
//!   use the same addresses.
//! - [`guarded_client_builder`] wires the resolver into a client that follows
//!   no redirects and ignores system proxies.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use super::provider::{ProviderConfig, ProviderType};
use crate::tools::limits::{self, Limits};

/// The provider hosts the hosted profile allows when no explicit allowlist is
/// configured.
pub const HOSTED_DEFAULT_LLM_HOSTS: &[&str] =
    &["api.openai.com", "api.anthropic.com", "openrouter.ai"];

/// The environment variable holding an explicit, comma-separated host list.
const ALLOWLIST_ENV: &str = "P2A_LLM_ALLOWED_HOSTS";

/// How long a DNS lookup of a provider host may take.
const LOOKUP_TIMEOUT: Duration = Duration::from_secs(10);

/// How long establishing a connection to a provider may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Which LLM provider endpoints this server connects to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgressPolicy {
    /// Normalised host names a `base_url` (and every resolved name) must
    /// match; `None` allows any public host.
    pub allowed_hosts: Option<Vec<String>>,
    /// Whether the Ollama default endpoint and other loopback Ollama
    /// endpoints may be used.
    pub allow_loopback: bool,
}

/// A refused egress: the host is not allowed, or resolves to an address the
/// server does not connect to. The message names no environment variable
/// and no resolved address.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct EgressError(pub String);

/// The raw allowlist from the environment, if set.
pub(crate) fn allowlist_from_env() -> Option<String> {
    std::env::var(ALLOWLIST_ENV).ok()
}

/// Lowercase, trimmed, without a trailing dot.
fn normalize_host(host: &str) -> String {
    let h = host.trim().to_ascii_lowercase();
    h.strip_suffix('.').map(str::to_string).unwrap_or(h)
}

impl EgressPolicy {
    /// Build the policy from the raw `P2A_LLM_ALLOWED_HOSTS` value and the
    /// active limits.
    ///
    /// Entries are trimmed, lowercased and stripped of a trailing `.`;
    /// empty entries are dropped, and a list with no entries counts as
    /// unset. Without an explicit list, the hosted default
    /// ([`HOSTED_DEFAULT_LLM_HOSTS`]) applies when the limits forbid
    /// loopback (the hosted floor).
    pub fn from_parts(env_allowlist: Option<&str>, limits: &Limits) -> Self {
        let explicit: Option<Vec<String>> = env_allowlist
            .map(|raw| {
                raw.split(',')
                    .map(normalize_host)
                    .filter(|h| !h.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|list| !list.is_empty());
        let allowed_hosts = explicit.or_else(|| {
            (!limits.llm_allow_loopback).then(|| {
                HOSTED_DEFAULT_LLM_HOSTS
                    .iter()
                    .map(|h| h.to_string())
                    .collect()
            })
        });
        Self {
            allowed_hosts,
            allow_loopback: limits.llm_allow_loopback,
        }
    }

    /// The policy from the environment and [`limits::current`].
    pub fn current() -> Self {
        Self::from_parts(allowlist_from_env().as_deref(), &limits::current())
    }

    /// Whether `host` passes the allowlist (always, without one).
    pub fn host_allowed(&self, host: &str) -> bool {
        match &self.allowed_hosts {
            None => true,
            Some(list) => {
                let host = normalize_host(host);
                list.contains(&host)
            }
        }
    }

    /// Vet a provider `base_url`: the allowlist first, then the scheme, then
    /// an IP-literal host. `http` and loopback hosts are accepted only for
    /// Ollama and only when the policy allows loopback.
    pub fn check_base_url(
        &self,
        provider: ProviderType,
        base_url: &str,
    ) -> Result<(), EgressError> {
        let url = reqwest::Url::parse(base_url)
            .map_err(|e| EgressError(format!("invalid base_url: {e}")))?;
        let raw_host = url
            .host_str()
            .ok_or_else(|| EgressError("base_url has no host".to_string()))?;
        // `host_str` keeps the brackets of an IPv6 literal; WHATWG parsing
        // has already normalised numeric IPv4 forms such as `2130706433`.
        let host = normalize_host(
            raw_host
                .strip_prefix('[')
                .and_then(|h| h.strip_suffix(']'))
                .unwrap_or(raw_host),
        );

        if !self.host_allowed(&host) {
            return Err(EgressError(format!(
                "base_url host `{host}` is not on this server's list of allowed LLM hosts"
            )));
        }

        let ip = host.parse::<IpAddr>().ok();
        let loopback_exempt = provider == ProviderType::Ollama
            && self.allow_loopback
            && (host == "localhost" || ip.is_some_and(|ip| ip.to_canonical().is_loopback()));

        match url.scheme() {
            "https" => {}
            "http" if loopback_exempt => {}
            other => {
                return Err(EgressError(format!(
                    "base_url scheme `{other}` is not allowed; use https \
                     (http is permitted only for a local Ollama endpoint)"
                )));
            }
        }

        if let Some(ip) = ip
            && is_disallowed_ip(ip)
            && !loopback_exempt
        {
            return Err(EgressError(format!(
                "base_url host {host} is in a private, loopback, link-local or reserved \
                 range and is not allowed"
            )));
        }
        Ok(())
    }

    /// Vet the URL a provider config will actually use: its `base_url` when
    /// set and non-blank, otherwise the provider's built-in default.
    pub fn check_provider_config(&self, cfg: &ProviderConfig) -> Result<(), EgressError> {
        let url = cfg
            .base_url
            .as_deref()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| cfg.provider_type.default_base_url());
        self.check_base_url(cfg.provider_type, url)
    }
}

/// True for an address the server must never connect to on a caller's
/// behalf: private, loopback, link-local, CGNAT, multicast, broadcast,
/// unspecified, documentation and other IANA special-purpose ranges, and the
/// IPv4 address embedded in IPv4-mapped, IPv4-compatible, NAT64, 6to4 and
/// Teredo IPv6 addresses.
pub fn is_disallowed_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4_disallowed(v4),
        IpAddr::V6(v6) => v6_disallowed(v6),
    }
}

fn v4_disallowed(v4: Ipv4Addr) -> bool {
    let o = v4.octets();
    v4.is_private()
        || v4.is_loopback()
        || v4.is_link_local()
        || v4.is_broadcast()
        || v4.is_unspecified()
        || v4.is_multicast()
        // 0.0.0.0/8 "this network"
        || o[0] == 0
        // 100.64.0.0/10 shared address space (CGNAT)
        || (o[0] == 100 && (o[1] & 0xc0) == 64)
        // 192.0.0.0/24 IETF protocol assignments
        || (o[0] == 192 && o[1] == 0 && o[2] == 0)
        // 198.18.0.0/15 benchmarking
        || (o[0] == 198 && (o[1] & 0xfe) == 18)
        // 240.0.0.0/4 reserved
        || o[0] >= 240
        // documentation: 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24
        || (o[0] == 192 && o[1] == 0 && o[2] == 2)
        || (o[0] == 198 && o[1] == 51 && o[2] == 100)
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)
}

fn v6_disallowed(v6: Ipv6Addr) -> bool {
    let s = v6.segments();
    let o = v6.octets();
    if v6.is_loopback() || v6.is_unspecified() {
        return true;
    }
    // fc00::/7 unique local, fe80::/10 link-local, fec0::/10 site-local,
    // ff00::/8 multicast, 2001:db8::/32 documentation, 64:ff9b:1::/48 local
    // NAT64.
    if (s[0] & 0xfe00) == 0xfc00
        || (s[0] & 0xffc0) == 0xfe80
        || (s[0] & 0xffc0) == 0xfec0
        || (s[0] & 0xff00) == 0xff00
        || (s[0] == 0x2001 && s[1] == 0x0db8)
        || (s[0] == 0x0064 && s[1] == 0xff9b && s[2] == 0x0001)
    {
        return true;
    }
    let low32 = Ipv4Addr::new(o[12], o[13], o[14], o[15]);
    // ::ffff:0:0/96 IPv4-mapped
    if let Some(v4) = v6.to_ipv4_mapped() {
        return v4_disallowed(v4);
    }
    // ::/96 IPv4-compatible (`::` and `::1` are handled above)
    if s[..6].iter().all(|&x| x == 0) {
        return v4_disallowed(low32);
    }
    // 64:ff9b::/96 NAT64
    if s[0] == 0x0064 && s[1] == 0xff9b && s[2..6].iter().all(|&x| x == 0) {
        return v4_disallowed(low32);
    }
    // 2002::/16 6to4: the IPv4 address is bits 16..48
    if s[0] == 0x2002 {
        return v4_disallowed(Ipv4Addr::new(o[2], o[3], o[4], o[5]));
    }
    // 2001:0::/32 Teredo: server in bits 32..64, client is the bitwise NOT of
    // bits 96..128
    if s[0] == 0x2001 && s[1] == 0 {
        let server = Ipv4Addr::new(o[4], o[5], o[6], o[7]);
        let client = Ipv4Addr::new(!o[12], !o[13], !o[14], !o[15]);
        return v4_disallowed(server) || v4_disallowed(client);
    }
    false
}

/// The DNS lookup a [`GuardedResolver`] runs: a host name to its addresses.
pub type Lookup =
    Arc<dyn Fn(String) -> BoxFuture<'static, std::io::Result<Vec<IpAddr>>> + Send + Sync>;

/// A `reqwest` DNS resolver that resolves a provider host and refuses it when
/// any answer is an address the server does not connect to.
pub struct GuardedResolver {
    policy: Arc<EgressPolicy>,
    loopback_ok: bool,
    lookup: Lookup,
}

impl GuardedResolver {
    /// A resolver using the system resolver (`tokio::net::lookup_host`).
    /// `loopback_ok` admits loopback answers (the Ollama exemption).
    pub fn new(policy: Arc<EgressPolicy>, loopback_ok: bool) -> Self {
        let lookup: Lookup = Arc::new(|host: String| {
            Box::pin(async move {
                let addrs = tokio::net::lookup_host((host.as_str(), 0)).await?;
                Ok(addrs.map(|a| a.ip()).collect())
            })
        });
        Self::with_lookup(policy, loopback_ok, lookup)
    }

    /// A resolver with an explicit lookup function (a test seam).
    pub fn with_lookup(policy: Arc<EgressPolicy>, loopback_ok: bool, lookup: Lookup) -> Self {
        Self {
            policy,
            loopback_ok,
            lookup,
        }
    }
}

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn egress(msg: String) -> BoxError {
    Box::new(EgressError(msg))
}

impl Resolve for GuardedResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = normalize_host(name.as_str());
        let policy = self.policy.clone();
        let loopback_ok = self.loopback_ok;
        let lookup = self.lookup.clone();
        Box::pin(async move {
            if !policy.host_allowed(&host) {
                return Err(egress(format!(
                    "LLM provider host `{host}` is not on this server's list of allowed LLM hosts"
                )));
            }
            let ips = match tokio::time::timeout(LOOKUP_TIMEOUT, lookup(host.clone())).await {
                Ok(Ok(ips)) => ips,
                Ok(Err(e)) => {
                    return Err(Box::new(std::io::Error::new(
                        e.kind(),
                        format!("LLM provider host `{host}` could not be resolved"),
                    )) as BoxError);
                }
                Err(_) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("resolving LLM provider host `{host}` timed out"),
                    )) as BoxError);
                }
            };
            if ips.is_empty() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("LLM provider host `{host}` did not resolve to any address"),
                )) as BoxError);
            }
            // Refuse the whole name when any answer is internal: filtering
            // would let a mixed answer pick the internal address later.
            if ips.iter().any(|ip| {
                is_disallowed_ip(*ip) && !(loopback_ok && ip.to_canonical().is_loopback())
            }) {
                return Err(egress(format!(
                    "LLM provider host `{host}` resolves to an address this server does not \
                     connect to"
                )));
            }
            let addrs: Vec<SocketAddr> = ips.into_iter().map(|ip| SocketAddr::new(ip, 0)).collect();
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

/// A client builder that resolves through `resolver`, follows no redirects,
/// ignores system proxies and gives up connecting after 10 s.
pub fn guarded_client_builder(resolver: GuardedResolver) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .dns_resolver(Arc::new(resolver))
}

/// The egress refusal in `err`'s source chain, if any.
pub(crate) fn find_egress_error(err: &reqwest::Error) -> Option<&EgressError> {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(e) = source {
        if let Some(egress) = e.downcast_ref::<EgressError>() {
            return Some(egress);
        }
        source = e.source();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const DENIED: &[&str] = &[
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "127.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "100.127.255.254",
        "0.0.0.0",
        "0.1.2.3",
        "255.255.255.255",
        "224.0.0.1",
        "240.0.0.1",
        "192.0.0.8",
        "198.18.0.1",
        "198.19.255.255",
        "192.0.2.1",
        "198.51.100.1",
        "203.0.113.1",
        "::1",
        "::",
        "fc00::1",
        "fd12:3456::1",
        "fe80::1",
        "fec0::1",
        "ff02::1",
        "2001:db8::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "::ffff:10.0.0.1",
        "::127.0.0.1",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b::7f00:1",
        "64:ff9b:1::1",
        "2002:7f00:1::1",
        "2002:a9fe:a9fe::1",
        // Teredo, client = !0x80fffffe = 127.0.0.1
        "2001:0:4136:e378:8000:63bf:80ff:fffe",
    ];
    const ALLOWED: &[&str] = &[
        "8.8.8.8",
        "1.1.1.1",
        "100.63.255.255",
        "100.128.0.1",
        "172.32.0.1",
        "198.20.0.1",
        "2606:4700:4700::1111",
        "::ffff:8.8.8.8",
        "64:ff9b::808:808",
        "2002:808:808::1",
        // Teredo, client = 8.8.8.8, server 65.54.227.120
        "2001:0:4136:e378:8000:63bf:f7f7:f7f7",
    ];

    #[test]
    fn disallowed_ip_table() {
        for s in DENIED {
            assert!(
                is_disallowed_ip(s.parse().unwrap()),
                "{s} must be disallowed"
            );
        }
        for s in ALLOWED {
            assert!(!is_disallowed_ip(s.parse().unwrap()), "{s} must be allowed");
        }
    }

    fn open() -> EgressPolicy {
        EgressPolicy {
            allowed_hosts: None,
            allow_loopback: true,
        }
    }

    #[test]
    fn literal_hosts_use_the_full_check() {
        for u in [
            "https://[::ffff:127.0.0.1]/",
            "https://[::ffff:169.254.169.254]/",
            "https://100.64.0.1/",
            "https://[64:ff9b::a9fe:a9fe]/",
            "https://[2002:a9fe:a9fe::1]/",
            "https://2130706433/",
            "https://0x7f.1/",
            "https://198.18.0.1/",
        ] {
            assert!(
                open().check_base_url(ProviderType::OpenAI, u).is_err(),
                "{u}"
            );
        }
        assert!(
            open()
                .check_base_url(ProviderType::Ollama, "http://[::ffff:127.0.0.1]:11434")
                .is_ok()
        );
        assert!(
            open()
                .check_base_url(ProviderType::Ollama, "http://10.0.0.5:11434")
                .is_err()
        );
    }

    #[test]
    fn hosted_profile_defaults_to_provider_allowlist() {
        let p = EgressPolicy::from_parts(None, &Limits::hosted());
        let want: Vec<String> = HOSTED_DEFAULT_LLM_HOSTS
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(p.allowed_hosts.as_ref(), Some(&want));
        assert!(!p.allow_loopback);
        assert!(
            p.check_base_url(ProviderType::OpenAI, "https://api.openai.com/v1")
                .is_ok()
        );
        assert!(
            p.check_base_url(ProviderType::OpenAI, "https://openrouter.ai/api/v1")
                .is_ok()
        );
        assert!(
            p.check_base_url(ProviderType::OpenAI, "https://evil.example/v1")
                .is_err()
        );
        // The default Ollama endpoint.
        assert!(p.check_provider_config(&ProviderConfig::default()).is_err());
        let off = EgressPolicy::from_parts(None, &Limits::off());
        assert_eq!(off, open());
        assert!(
            off.check_provider_config(&ProviderConfig::default())
                .is_ok()
        );
    }

    #[test]
    fn explicit_allowlist_revokes_default_ollama_endpoint() {
        let p = EgressPolicy::from_parts(Some(" API.OpenAI.com, ,openrouter.ai "), &Limits::off());
        assert_eq!(
            p.allowed_hosts,
            Some(vec![
                "api.openai.com".to_string(),
                "openrouter.ai".to_string()
            ])
        );
        assert!(p.check_provider_config(&ProviderConfig::default()).is_err());
        assert!(
            p.check_base_url(ProviderType::OpenAI, "https://api.openai.com./v1")
                .is_ok()
        );
    }

    #[test]
    fn refusals_name_no_environment_variables() {
        let e = EgressPolicy::from_parts(None, &Limits::hosted())
            .check_base_url(ProviderType::OpenAI, "https://evil.example/v1")
            .unwrap_err()
            .to_string();
        assert!(!e.contains("P2A_"), "{e}");
    }
}
