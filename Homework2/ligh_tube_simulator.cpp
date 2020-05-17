//=============================================================================================
// Computer Graphics 2nd Homework - Light tube simulator (Spring 2020)
//=============================================================================================
#include "framework.h"

enum  MaterialType { ROUGH, REFLECTIVE };

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3 _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

struct Ellipsoid : public Intersectable {
	vec3 center;
	vec3 param;
	float upperLimit;

	Ellipsoid(const vec3 _center, vec3 _param, float _upperLimit, Material* _material) {
		center = _center;
		param = _param;
		upperLimit = _upperLimit;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = ((ray.dir.x * ray.dir.x) / (param.x * param.x)) + ((ray.dir.y * ray.dir.y) / (param.y * param.y)) + ((ray.dir.z * ray.dir.z) / (param.z * param.z));
		float b = (((dist.x * ray.dir.x) / (param.x * param.x)) + ((dist.y * ray.dir.y) / (param.y * param.y)) + ((dist.z * ray.dir.z) / (param.z * param.z))) * 2.0f;
		float c = (((dist.x * dist.x) / (param.x * param.x)) + ((dist.y * dist.y) / (param.y * param.y)) + ((dist.z * dist.z) / (param.z * param.z))) - 1.0f;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		vec3 upp = vec3(0.0f, 0.0f, upperLimit);
		vec3 uppN = normalize(vec3(0.0f, 0.0f, upperLimit));
		if (dot(uppN, ray.start + t1 * ray.dir - upp) > 0.0f) {
			t1 = -1;
		}
		if (dot(uppN, ray.start + t2 * ray.dir - upp) > 0.0f) {
			t2 = -1;
		}
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 normal = hit.position - center;
		normal.x = 2.0f * normal.x / (param.x * param.x);
		normal.y = 2.0f * normal.y / (param.y * param.y);
		normal.z = 2.0f * normal.z / (param.z * param.z);
		hit.normal = normalize(normal);
		hit.material = material;
		return hit;
	}
};

struct ElliticalCylinder : public Intersectable {
	vec3 center;
	vec3 param;
	vec3 lowerLimit;
	float upperLimit;

	ElliticalCylinder(const vec3 _center, vec3 _param, float _lowerLimit, float _upperLimit, Material* _material) {
		center = _center;
		param = _param;
		lowerLimit = _lowerLimit;
		upperLimit = _upperLimit;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = ((ray.dir.x * ray.dir.x) / (param.x * param.x)) + ((ray.dir.y * ray.dir.y) / (param.y * param.y));
		float b = (((dist.x * ray.dir.x) / (param.x * param.x)) + ((dist.y * ray.dir.y) / (param.y * param.y))) * 2.0f;
		float c = (((dist.x * dist.x) / (param.x * param.x)) + ((dist.y * dist.y) / (param.y * param.y))) - 1.0f;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		vec3 upp = vec3(0.0f, 0.0f, upperLimit);
		vec3 uppN = normalize(vec3(0.0f, 0.0f, upperLimit));
		if (dot(uppN, ray.start + t1 * ray.dir - upp) > 0) {
			t1 = -1;
		}
		if (dot(uppN, ray.start + t2 * ray.dir - upp) > 0) {
			t2 = -1;
		}
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 normal = hit.position - center;
		normal.x = 2.0f * normal.x / (param.x * param.x);
		normal.y = 2.0f * normal.y / (param.y * param.y);
		normal.z = 0.0;
		hit.normal = normalize(normal);
		hit.material = material;
		return hit;
	}
};

struct Paraboloid : public Intersectable {
	vec3 center;
	vec3 size;

	Paraboloid(const vec3 _center, vec3 _size, Material* _material) {
		center = _center;
		size = _size;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = ((ray.dir.x * ray.dir.x) / (size.x * size.x)) + ((ray.dir.y * ray.dir.y) / (size.y * size.y));
		float b = ((2.0f * dist.x * ray.dir.x) / (size.x * size.x)) + ((2.0f * dist.y * ray.dir.y) / (size.y * size.y)) + ray.dir.z;
		float c = (((dist.x * dist.x) / (size.x * size.x)) + ((dist.y * dist.y) / (size.y * size.y)) + (dist.z));
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 normal = hit.position - center;
		normal.x = 2.0f * normal.x / (size.x * size.x);
		normal.y = 2.0f * normal.y / (size.y * size.y);
		normal.z = 1.0f;
		hit.normal = normalize(normal);
		hit.material = material;
		return hit;
	}
};

struct Hyperboloid : public Intersectable {
	vec3 center;
	vec3 size;
	float lowerLimit;
	float upperLimit;

	Hyperboloid(const vec3 _center, vec3 _size, float _lowerLimit, float _upperLimit, Material* _material) {
		center = _center;
		size = _size;
		lowerLimit = _lowerLimit;
		upperLimit = _upperLimit;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = ((ray.dir.x * ray.dir.x) / (size.x * size.x)) + ((ray.dir.y * ray.dir.y) / (size.y * size.y)) - ((ray.dir.z * ray.dir.z) / (size.z * size.z));
		float b = (((dist.x * ray.dir.x) / (size.x * size.x)) + ((dist.y * ray.dir.y) / (size.y * size.y)) - ((dist.z * ray.dir.z) / (size.z * size.z))) * 2.0f;
		float c = (((dist.x * dist.x) / (size.x * size.x)) + ((dist.y * dist.y) / (size.y * size.y)) - ((dist.z * dist.z) / (size.z * size.z))) - 1.0f;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		vec3 low = vec3(0.0f, 0.0f, lowerLimit);
		vec3 lowN = normalize(vec3(0.0f, 0.0f, lowerLimit));
		vec3 upp = vec3(0.0f, 0.0f, upperLimit);
		vec3 uppN = normalize(vec3(0.0f, 0.0f, upperLimit));
		if (dot(uppN, ray.start + t1 * ray.dir - upp) > 0.0f || (dot(lowN, ray.start + t1 * ray.dir - low) < 0.0f)) {
			t1 = -1;
		}
		if (dot(uppN, ray.start + t2 * ray.dir - upp) > 0.0f || (dot(lowN, ray.start + t2 * ray.dir - low) < 0.0f)) {
			t2 = -1;
		}
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec3 normal = hit.position - center;
		normal.x = 2.0f * normal.x / (size.x * size.x);
		normal.y = 2.0f * normal.y / (size.y * size.y);
		normal.z = -2.0f * normal.z / (size.z * size.z);
		hit.normal = normalize(normal);
		hit.material = material;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<vec3> testPoints;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
	vec3 Sky;
	float radius = 0.398f;
public:
	void build() {
		vec3 eye = vec3(1.8f, 0.0f, 0.2f), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0.0);
		float fov = 85 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.1f);
		Sky = vec3(0.4f, 0.4f, 0.7f);

		vec3 lightDirection(0, 0, 5), Le(15, 15, 15);
		lights.push_back(new Light(lightDirection, Le));

		vec3 ks(2, 2, 2);
		Material* material = new RoughMaterial(vec3(0.3f, 0.2f, 0.1f), vec3(0.5f, 0.5f, 0.5f), 50);
		Material* blueMaterial = new RoughMaterial(vec3(0.1f, 0.2f, 0.3f), ks, 50);
		Material* greenMaterial = new RoughMaterial(vec3(0.0f, 0.15f, 0.0f), ks, 50);
		Material* whiteMaterial = new RoughMaterial(vec3(1.0f, 1.0f, 1.0f), ks, 50);
		Material* goldMaterial = new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
		Material* silverMaterial = new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));

		objects.push_back(new Ellipsoid(vec3(0.0f, 0.0f, 0.0f), vec3(2.0f, 2.0f, 1.0f), 0.98f, material));
		objects.push_back(new Hyperboloid(vec3(0.5f, -0.9f, 0.0f), vec3(0.1f, 0.1f, 0.4f), 0.0f, 0.4f, greenMaterial));
		objects.push_back(new ElliticalCylinder(vec3(-0.8f, -0.7f, 0.0f), vec3(0.4f, 0.4f, 0.0f), 0.0f, 0.7f, blueMaterial));
		objects.push_back(new Paraboloid(vec3(-0.6f, 1.1f, 0.4f), vec3(0.8f, 0.8f, 0.8f), goldMaterial));
		objects.push_back(new Hyperboloid(vec3(0.0f, 0.0f, 0.98f), vec3(0.398f, 0.398f, 0.5f), 0.98f, 10.0f, silverMaterial));

		for (int i = 0; i < 10; i++) {
			float r = radius * sqrtf(((float)rand() / RAND_MAX));
			float theta = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
			float x = r * cosf(theta);
			float y = r * sinf(theta);
			testPoints.push_back(vec3(x, y, 0.98f));
		}

	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) {
			return Sky + lights[0]->Le * powf(dot(ray.dir, lights[0]->direction), 10);
		}

		Hit hit = firstIntersect(ray);
		if (hit.t < 0) {
			return Sky + lights[0]->Le * powf(dot(ray.dir, lights[0]->direction), 10);
		}

		vec3 outRadiance(0, 0, 0);
		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (int i = 0; i < testPoints.size(); i++) {
				for (Light* light : lights) {
					vec3 dir = normalize(testPoints[i] - hit.position);
					Ray shadowRay(hit.position + hit.normal * epsilon, dir);
					float cosTheta = dot(hit.normal, dir);
					float cosTheta2 = -dot(vec3(0.0f, 0.0f, -1.0f), normalize(dir));
					if (cosTheta > 0) {
						vec3 outRadiance2 = trace(shadowRay, depth + 1);
						float omega = (radius * radius * M_PI) / (testPoints.size()) * ((cosTheta2) / (length(dir) * length(dir)));
						outRadiance = outRadiance + outRadiance2 * hit.material->kd * cosTheta * omega;
						vec3 halfway = normalize(-ray.dir + dir);
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0) {
							outRadiance = outRadiance + outRadiance2 * hit.material->ks * powf(cosDelta, hit.material->shininess) * omega;
						}
					}
				}
			}
		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}

		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao = 0, textureId = 0;	// vertex array object id and texture id
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glGenTextures(1, &textureId);				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);	// binding
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId); //binding
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]); // To GPU
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}
void onIdle() {}