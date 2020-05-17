//=============================================================================================
// Computer Graphics 3rd Homework - Virus killing antibody (Spring 2020)
//=============================================================================================
#include "framework.h"

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d; // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 20;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 10;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

//---------------------------
class AntiBodyTexture : public Texture {
	//---------------------------
public:
	AntiBodyTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 purple(0.8f, 0.8f, 0.8f, 0.2f);
		for (int x = 0; x < width; x++)
			for (int y = 0; y < height; y++) {
				image[y * width + x] = purple;
			}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
class CoronaSpikeTexture : public Texture {
	//---------------------------
public:
	CoronaSpikeTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 red(0.5f, 0, 0, 0.5f);
		for (int x = 0; x < width; x++)
			for (int y = 0; y < height; y++) {
				image[y * width + x] = red;
			}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
class SphereTexture : public Texture {
	//---------------------------
public:
	SphereTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
class VirusSphereTexture : public Texture {
	//---------------------------
public:
	VirusSphereTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 gray(0.5f, 0.5f, 0.0f, 0.2f);
		for (int x = 0; x < width; x++)
			for (int y = 0; y < height; y++) {
				image[y * width + x] = gray;
			}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4				MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light>	lights;
	Texture* texture;
	vec3				wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class GouraudShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;		 // number of light sources
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   if (dot(N, V) < 0) N = -N;	//prepare for one-sided surfaces like Mobius or Klein
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniform(state.lights[0].wLightPos, "wLightPos");
	}
};

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	float R = 1.0f;
	float tetraR;
	std::vector<VertexData> vtxData;
	std::vector<VertexData> spikePointData;
	std::vector<VertexData> vtxPoints;

	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	virtual void Animate() {}
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class TriangleMesh : public Geometry {
	//---------------------------
public:
	std::vector<VertexData> mesh;

	void Create() {
		glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(VertexData), &mesh[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, mesh.size());
	}
};

//---------------------------
struct Tetrahedron : public TriangleMesh {
	//---------------------------
	void PushTriangle(VertexData vd[3]) {
		mesh.push_back(vd[0]); mesh.push_back(vd[1]); mesh.push_back(vd[2]);
	}
	float height = 1.0f;
	bool max = false;
	Tetrahedron() {
		Generate();
		Create();
	}
	void Generate(vec3 _p1 = vec3(1, 1, 1), vec3 _p2 = vec3(-1, -1, 1), vec3 _p3 = vec3(-1, 1, -1), int actualLevel=0, vec3 p = vec3(0,0,0)) {
		if (actualLevel > 2) return;
			vec3 p1 = _p1, p2 = _p2, p3 = _p3;
			vec3 normal, p4;
			if (actualLevel == 0) {
				normal = normalize(cross(p1 - p2, p1 - p3));
				p4 = (p1 + p2 + p3) / 3 + normal * sqrtf((length(p1 - p3) * length(p1 - p3)) - (length((p1 - p2) / 2) * length((p1 - p2) / 2)));
				tetraR = length(((p1 + p2 + p3 + p4) / 4) - p1);
			}
			else {
				normal = normalize(cross(p1 - p2, p1 - p3))*height;
				if (length(p - normal) > length(p - -normal)) {
					normal = -normal;
				}
				p4 = (p1 + p2 + p3) / 3 + normal * sqrtf((length(p1 - p3) * length(p1 - p3)) - (length((p1 - p2) / 2) * length((p1 - p2) / 2)));
			}
			VertexData vd[3];
			vd[0].texcoord = vec2(0, 0); vd[1].texcoord = vec2(1, 0); vd[2].texcoord = vec2(1, 1);

			if (actualLevel == 0) {
				vd[0].position = p1; vd[1].position = p2; vd[2].position = p3;
				vd[0].normal = vd[1].normal = vd[2].normal = cross(p1 - p2, p1 - p3);
				PushTriangle(vd);
				Generate((p1 + p2) / 2, (p1 + p3) / 2, (p3 + p2) / 2, actualLevel + 1, -(p1 + p2 + p3 + p4) / 4);
			}

			vd[0].position = p1; vd[1].position = p2; vd[2].position = p4;
			vd[0].normal = vd[1].normal = vd[2].normal = cross(p1 - p2, p1 - p4);
			PushTriangle(vd);
			Generate((p1 + p2) / 2, (p1 + p4) / 2, (p2 + p4) / 2, actualLevel + 1, (p1 + p2 + p3 + p4) / 4);

			vd[0].position = p1; vd[1].position = p3; vd[2].position = p4;
			vd[0].normal = vd[1].normal = vd[2].normal = cross(p1 - p3, p1 - p4);
			PushTriangle(vd);
			Generate((p1 + p3) / 2, (p1 + p4) / 2, (p3 + p4) / 2, actualLevel + 1, (p1 + p2 + p3 + p4) / 4);

			vd[0].position = p2; vd[1].position = p3; vd[2].position = p4;
			vd[0].normal = vd[1].normal = vd[2].normal = cross(p2 - p3, p2 - p4);
			PushTriangle(vd);
			Generate((p2 + p3) / 2, (p2 + p4) / 2, (p3 + p4) / 2, actualLevel + 1, (p1 + p2 + p3 + p4) / 4);
	}
	void Animate() override {
		if (!max) {
			height += 0.003f;
			if (height >= 2.0f) {
				max = true;
			}
		}
		if(max){
			height -= 0.003f;
			if (height <= 1.0f) {
				max = false;
			}
		}
		mesh.clear();
		Generate();
		Create();
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	virtual VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		int n = 20;
		for (int i = 0; i < n; i+=2) {
			float theta = M_PI * ((float) i / n );
			int numM = n * sinf(theta);
			for (int j = 0; j <= numM; j++) {
				 spikePointData.push_back(GenVertexData((float)j / numM , (float)i / n));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);
	}
};

//---------------------------
class CoronaSphere : public ParamSurface {
	//---------------------------
public:
	CoronaSphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);
		X.f = X.f * R;
		Y.f = Y.f * R;
		Z.f = Z.f * R;
	}
};

//---------------------------
class Tractricoid : public ParamSurface {
	//---------------------------
public:
	const float height = 3.0f;
	Tractricoid() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		
		U = U * height, V = V * 2 * M_PI;
		X = Cos(V) / Cosh(U); Y = Sin(V) / Cosh(U); Z = U - Tanh(U);
	}
};

//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	bool movable = false;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	virtual void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) {
		rotationAngle = 0.8f * tend;
		geometry->Animate();
	}
};

//---------------------------
struct SpikeObject : Object {
	//---------------------------
	Object* parent;

public:
	SpikeObject(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry, Object* _parent )
		: Object(_shader, _material, _texture, _geometry) 
	{
		parent = _parent;
	}
	void SetModelingTransform(mat4& M, mat4& Minv) override {

		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation) 
			* ScaleMatrix(parent->scale*1.15f) * RotationMatrix(parent->rotationAngle, parent->rotationAxis) * TranslateMatrix(parent->translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z))
			* TranslateMatrix(-parent->translation) * RotationMatrix(-parent->rotationAngle, parent->rotationAxis) * ScaleMatrix(vec3(1 / parent->scale.x*1.15f, 1 / parent->scale.y*1.15f, 1 / parent->scale.z*1.15f));
	}
	void Draw(RenderState state) override{
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	void Animate(float tstart, float tend) override { }
};

float rnd() { return (float)rand() / RAND_MAX; }

struct AntiObject : Object {
public:
	AntiObject(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) : Object(_shader, _material, _texture, _geometry) {}

	void Animate(float tstart, float tend) override {
		rotationAngle = 0.8f * tend;
		geometry->Animate();
		if (movable) {
			float limit = 0.05f;
			vec3 randomDirection = vec3(rnd() * limit - limit / 2, rnd() * limit - limit / 2, rnd() * limit - limit / 2);
			translation = translation +  randomDirection;
		}
	}
};

struct CoronaObject : Object {
public:
	CoronaObject(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) : Object(_shader, _material, _texture, _geometry) {}

	void Animate(float tstart, float tend) override {
		if (movable) {
			rotationAngle = 0.8f * tend;
			float limit = 0.02f;
			vec3 randomDirection = vec3(rnd() * limit - limit / 2, rnd() * limit - limit / 2, rnd() * limit - limit / 2);
			translation = translation + randomDirection;
		}
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	Camera camera; // 3D camera
	std::vector<Light> lights;
	Object* antibody;
	Object* virus;
public:
	void Build() {
		// Shaders
		Shader* phongShader = new PhongShader();
		Shader* gouraudShader = new GouraudShader();
		Shader* nprShader = new NPRShader();

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		// Textures
		Texture* sphereTexture = new SphereTexture(15, 20);
		Texture* virusSphereTexture = new VirusSphereTexture(15, 20);
		Texture* coronaSpikeTexture = new CoronaSpikeTexture(15, 20);
		Texture* antiBodyTexture = new AntiBodyTexture(15, 20);

		// Geometries
		Geometry* sphere = new Sphere();
		Geometry* coronaSphere = new CoronaSphere();
		Geometry* tractricoid = new Tractricoid();
		Geometry* tetrahedron = new Tetrahedron();


		// Create objects by setting up their vertex data on the GPU
		Object* sphereObject0 = new Object(phongShader, material0, sphereTexture, sphere);
		sphereObject0->translation = vec3(0, 0, 0);
		sphereObject0->scale = vec3(10.0f, 10.0f, 10.0f);
		objects.push_back(sphereObject0);

		// Create objects by setting up their vertex data on the GPU
		virus = new CoronaObject(phongShader, material1, virusSphereTexture, coronaSphere);
		virus->translation = vec3(-1.5f, 0, 0);
		virus->rotationAxis = vec3(1, 0, 0);
		virus->rotationAngle = M_PI/2;
		virus->scale = vec3(1.0f, 1.0f, 1.0f);
		virus->movable = true;
		objects.push_back(virus);
		
		for (int i = 0; i < virus->geometry-> spikePointData.size(); i++) {
			vec3 normalvektor = virus->geometry-> spikePointData[i].normal;
			vec3 tractriiranyvektor = vec3(0, 0, 1);
			// Create objects by setting up their vertex data on the GPU
			SpikeObject* coronaSpike = new SpikeObject(phongShader, material1, coronaSpikeTexture, tractricoid, virus);
			coronaSpike->translation = virus->geometry-> spikePointData[i].position;
			coronaSpike->rotationAxis = cross(normalvektor, tractriiranyvektor);
			coronaSpike->rotationAngle = (virus->geometry-> spikePointData[i].texcoord.y) * M_PI + M_PI;
			coronaSpike->scale = vec3(0.1f, 0.1f, 0.1f);
			objects.push_back(coronaSpike);
		}

		antibody = new AntiObject(phongShader, material0, antiBodyTexture, tetrahedron);
		antibody->translation = vec3(1.5f, 0, 0);
		antibody->rotationAxis = vec3(0, 1, 0);
		antibody->scale = vec3(0.4f, 0.4f, 0.4f);
		antibody->movable = true;
		objects.push_back(antibody);


		// Camera
		camera.wEye = vec3(0, 0, 5);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);
		camera.bp = 15.0f;

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4(5, 5, 4, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0.3f, 0.3f, 1);
		lights[0].Le = vec3(3, 0, 0);

		lights[1].wLightPos = vec4(5, 10, 20, 0);	// ideal point -> directional light source
		lights[1].La = vec3(0.6f, 0.6f, 0.6f);
		lights[1].Le = vec3(0, 3, 0);

		lights[2].wLightPos = vec4(-5, 5, 5, 0);	// ideal point -> directional light source
		lights[2].La = vec3(0.3f, 0.3f, 0.3f);
		lights[2].Le = vec3(0, 0, 3);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		for (Object* obj : objects) obj->Animate(tstart, tend);
		if (length((antibody->translation) - (virus->translation)) <= (antibody->geometry->tetraR*antibody->scale.x)+ (virus->geometry->R * virus->scale.x)) {
			virus->movable = false;
		}
	}

	void Move(unsigned char key) {
		float length = 0.1f;
		if (key == 'x') if (rnd() < 0.5f) antibody->translation = antibody->translation + vec3(-length, 0, 0);
		if (key == 'X') if (rnd() < 0.5f) antibody->translation = antibody->translation + vec3(length, 0, 0);
		if (key == 'y') if (rnd() < 0.5f) antibody->translation = antibody->translation + vec3(0, -length, 0);
		if (key == 'Y') if (rnd() < 0.5f) antibody->translation = antibody->translation + vec3(0, length, 0);
		if (key == 'z') if (rnd() < 0.5f) antibody->translation = antibody->translation + vec3(0, 0, -length);
		if (key == 'Z') if (rnd() < 0.5f) antibody->translation = antibody->translation + vec3(0, 0, length);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
	scene.Move(key);
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}