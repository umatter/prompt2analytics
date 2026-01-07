export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set(["favicon.png"]),
	mimeTypes: {".png":"image/png"},
	_: {
		client: {start:"_app/immutable/entry/start.CkLSMqVZ.js",app:"_app/immutable/entry/app.CyR7DWR7.js",imports:["_app/immutable/entry/start.CkLSMqVZ.js","_app/immutable/chunks/Dxe1-o2_.js","_app/immutable/chunks/D0JSkPKu.js","_app/immutable/chunks/iOUKcL_x.js","_app/immutable/entry/app.CyR7DWR7.js","_app/immutable/chunks/D0JSkPKu.js","_app/immutable/chunks/DxAPyo36.js","_app/immutable/chunks/eaGfzuTu.js","_app/immutable/chunks/iOUKcL_x.js","_app/immutable/chunks/BWpIWWid.js","_app/immutable/chunks/DhWLDm4_.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
