//=============================================================================================
// Computer Graphics 1st Homework - Sirius triangle drawing (Spring 2020)
//=============================================================================================
#include "framework.h"

const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers
 
	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
 
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel
 
	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram;
const int nTesselatedVertices = 100;

class Circle
{
	unsigned int vao;
public:
	void create()
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		vec2 vertices[nTesselatedVertices];
		for (int i = 0; i < nTesselatedVertices; i++)
		{
			float fi = i * 2 * M_PI / nTesselatedVertices;
			vertices[i] = vec2(cosf(fi), sinf(fi));
		}
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * nTesselatedVertices, vertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	void Draw()
	{
		mat4 MVPTransform = { 1, 0, 0, 0,
							  0, 1, 0, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1 };
		gpuProgram.setUniform(MVPTransform, "MVP");
		gpuProgram.setUniform(vec3(0.5, 0.5, 0.5), "color");

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, nTesselatedVertices);
	}
};

class Curve
{
	unsigned int vaoVectorizedCurve, vboVectorizedCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	unsigned int vaoTriangles, vboTriangles;
protected:
	std::vector<vec2> wCtrlPoints;
	std::vector<vec2> vertexData;
	std::vector<vec2> triangleData;
	std::vector<vec2> centerData;
public:
	Curve()
	{
		glGenVertexArrays(1, &vaoVectorizedCurve);
		glBindVertexArray(vaoVectorizedCurve);
		glGenBuffers(1, &vboVectorizedCurve);
		glBindBuffer(GL_ARRAY_BUFFER, vboVectorizedCurve);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);

		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);
		glGenBuffers(1, &vboCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);

		glGenVertexArrays(1, &vaoTriangles);
		glBindVertexArray(vaoTriangles);
		glGenBuffers(1, &vboTriangles);
		glBindBuffer(GL_ARRAY_BUFFER, vboTriangles);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
	}
	~Curve()
	{
		glDeleteBuffers(1, &vboCtrlPoints);
		glDeleteVertexArrays(1, &vaoCtrlPoints);
		glDeleteBuffers(1, &vboVectorizedCurve);
		glDeleteVertexArrays(1, &vaoVectorizedCurve);
	}

	void AddControlPoint(float cX, float cY)
	{
		if (wCtrlPoints.size() < 3)
		{
			vec4 wVertex = vec4(cX, cY, 0, 1);
			wCtrlPoints.push_back(vec2(wVertex.x, wVertex.y));
		}
	}

	void Draw()
	{
		mat4 MVPTransform = { 1, 0, 0, 0,
							  0, 1, 0, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1 };
		gpuProgram.setUniform(MVPTransform, "MVP");

		//AddControlPoint();

		if (wCtrlPoints.size() > 0)
		{
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * sizeof(vec2), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 0, 0), "color");
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}

		if (wCtrlPoints.size() == 3)
		{
			int db = 0;
			for (int z = 0; z < 3; z++)
			{
				int idx = z + 1;
				if (idx == 3)
				{
					idx = 0;
				}

				vec2 p1 = vec2(wCtrlPoints[z].x, wCtrlPoints[z].y);
				vec2 p2 = vec2(wCtrlPoints[idx].x, wCtrlPoints[idx].y);

				float y = ((2 * (p2.x * ((p1.x * p1.x) + (p1.y * p1.y) + 1)) - (2 * (p1.x * ((p2.x * p2.x) + (p2.y * p2.y) + 1)))) / (4 * ((p2.x * p1.y) - (p1.x * p2.y))));
				float x = (((p1.x * p1.x) + (p1.y * p1.y) - (2 * p1.y * y) + 1.0f) / (2.0f * p1.x));

				float r = sqrtf(((x * x) + (y * y)) - 1);

				vec2 c = vec2(x, y);
				centerData.push_back(c);

				vec2 cp1 = p1 - c;
				vec2 cp2 = p2 - c;

				float polar1 = atan2(cp1.y, cp1.x);
				float polar2 = atan2(cp2.y, cp2.x);

				float diff = polar2 - polar1;
				if (polar1 > 0.0f && diff < -M_PI)
				{
					diff = ((2.0f * M_PI) - (polar1 + fabs(polar2)));
				}
				else if (polar1 < 0.0f && diff > M_PI)
				{
					diff = -((2.0f * M_PI) - (fabs(polar1) + polar2));
				}

				for (int i = 0; i < nTesselatedVertices; i++)
				{
					float fi = polar1 + i * (diff / nTesselatedVertices);
					vertexData.push_back(vec2(c.x + r * cosf(fi), c.y + r * sinf(fi)));
					db++;
				}
			}

			float angles[3];
			float angleSum = 0.0f;
			for (int i = 0; i < 3; i++)
			{
				int idx = i + 1;
				if (idx == 3)
				{
					idx = 0;
				}
				vec2 c1 = centerData[i];
				vec2 c2 = centerData[idx];
				vec2 p = wCtrlPoints[idx];
				vec2 cp1 = c1 - p;
				vec2 cp2 = c2 - p;
				vec2 cp1rot = vec2(cp1.y, -cp1.x);
				vec2 cp2rot = vec2(cp2.y, -cp2.x);
				float angle = acosf((dot(cp1rot, cp2rot) / (length(cp1rot) * length(cp2rot))));
				angle = angle * 180.0f / M_PI;
				if (angle > 90) angle = 180 - angle;
				angles[i] = angle;
				angleSum += angle;
			}
			printf("Alpha: %f, Beta: %f, Gamma: %f, Angle sum: %f\n", angles[1], angles[0], angles[2], angleSum);

			float lengths[3];
			for (int i = 0; i < 3; i++) {
				float lengthSum = 0.0f;
				for (int j = i * nTesselatedVertices; j < (i + 1) * nTesselatedVertices - 1; j++) {
					float sx = vertexData[j].x;
					float sy = vertexData[j].y;
					float dx = vertexData[j + 1].x - sx;
					float dy = vertexData[j + 1].y - sy;
					float ds = sqrtf((dx * dx) + (dy * dy)) / (1.0f - (sx * sx) - (sy * sy));
					lengthSum += ds;
				}
				lengths[i] = lengthSum;
			}
			printf("a: %f b: %f c: %f\n", lengths[1], lengths[0], lengths[2]);

			int db2 = 0;
			std::vector<vec2> pointData = vertexData;
			for (unsigned int i = 0; i < pointData.size() - 2; i++) {
				float x11 = pointData[i].x;
				float y11 = pointData[i].y;
				float x12 = pointData[i + 2].x;
				float y12 = pointData[i + 2].y;
				for (unsigned int j = 0; j < pointData.size() - 1; j++) {
					float x21 = pointData[j].x;
					float y21 = pointData[j].y;
					float x22 = pointData[j + 1].x;
					float y22 = pointData[j + 1].y;
					float t2 = (((y22 - y12) / (y11 - y12)) - ((x22 - x12) / (x11 - x12))) / (((x21 - x22) / (x11 - x12)) - ((y21 - y22) / (y11 - y12)));
					float t1 = (t2 * (x21 - x22) + (x22 - x12)) / (x11 - x12);
					if ((t1 <= 0.0f || t1 >= 1.0f) && (t2 <= 0.0f || t2 >= 1.0f)) {
						int intersection = 0;
						for (unsigned int k = 0; k < pointData.size() - 1; k++) {
							float x11f = (x11 + x12) / 2.0f;
							float y11f = (y11 + y12) / 2.0f;
							float x12f = 1.0f;
							float y12f = 1.0f;
							float x21k = pointData[k].x;
							float y21k = pointData[k].y;
							float x22k = pointData[k + 1].x;
							float y22k = pointData[k + 1].y;
							float t22 = (((y22k - y12f) / (y11f - y12f)) - ((x22k - x12f) / (x11f - x12f))) / (((x21k - x22k) / (x11f - x12f)) - ((y21k - y22k) / (y11f - y12f)));
							float t11 = (t22 * (x21k - x22k) + (x22k - x12f)) / (x11f - x12f);
							if ((t11 > 0.0f && t11 < 1.0f) && (t22 > 0.0f && t22 < 1.0f)) {
								intersection++;
							}
						}
						if (intersection % 2 == 1) {
							triangleData.push_back(pointData[i]);
							triangleData.push_back(pointData[i + 1]);
							triangleData.push_back(pointData[i + 2]);
							db2 += 3;
							pointData.erase(pointData.begin() + i + 1);
							if (i != 0) i--;
						}
					}
				}
			}


			glBindVertexArray(vaoTriangles);
			glBindBuffer(GL_ARRAY_BUFFER, vboTriangles);
			glBufferData(GL_ARRAY_BUFFER, triangleData.size() * sizeof(vec2), &triangleData[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(0, 0, 1), "color");
			glPointSize(2.0f);
			glDrawArrays(GL_TRIANGLES, 0, db2);

			glBindVertexArray(vaoVectorizedCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboVectorizedCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(vec2), &vertexData[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 1, 0), "color");
			glLineWidth(2.0f);
			glDrawArrays(GL_LINE_LOOP, 0, db);
		}
	}
};

Curve* curve;
Circle circle;

void onInitialization()
{
	glViewport(0, 0, windowWidth, windowHeight);

	curve = new Curve();
	circle.create();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay()
{
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	circle.Draw();
	curve->Draw();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		curve->AddControlPoint(cX, cY);
	}
}

void onIdle() {}