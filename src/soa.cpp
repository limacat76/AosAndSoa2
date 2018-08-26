#include <iostream>
#include <cmath>
#include <chrono>
#include <string>

// SSE Intrinsics
#include <emmintrin.h>
#include <mmintrin.h>

using std::chrono::system_clock;

struct Quadratic {
public:
	float a;
	float b;
	float c;
};

struct SoAQuadratic {
	float* a;
	float* b;
	float* c;
	SoAQuadratic(int size) : a(new float[size]),
		b(new float[size]),
		c(new float[size]) {
		;
	}
	~SoAQuadratic() {
		delete[] a;
		delete[] b;
		delete[] c;
	}
 };



void printTime(const std::string& what, system_clock::time_point start, system_clock::time_point finish) {
	std::cout << what << " took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
		<< "us.\n";
}


int main() {
	system_clock::time_point start;
	system_clock::time_point finish;

	system_clock::time_point aosStart;
	system_clock::time_point aosFinish;

	system_clock::time_point soaStart;
	system_clock::time_point soaFinish;

	const int count = 1 << 24;
	Quadratic* e = new Quadratic[count]; // Array of Structures
	SoAQuadratic soaE(count); // Struct of Arrays
	float* roots = new float[count]; // this will be our roots
	float* rootsAos = new float[count]; // this will be our roots

	start = system_clock::now();

	// Set up random quadratics
	for (int i = 0; i < count; i++) {
		e[i].a = (rand() % 100) - 50;
		soaE.a[i] = e[i].a;
		e[i].b = (rand() % 100) - 50;
		soaE.b[i] = e[i].b;
		e[i].c = (rand() % 100) - 50;
		soaE.c[i] = e[i].c;
	}
	finish = system_clock::now();
	printTime("Construction", start, finish);

	start = system_clock::now();
	aosStart = start;
	for (int i = 0; i < count; i++) {
		roots[i] = (-e[i].b + sqrt(e[i].b*e[i].b - 4 * (e[i].a
			* e[i].c))) / (2.0f*e[i].a);
	}
	finish = system_clock::now();
	aosFinish = finish;
	printTime("Calculation", start, finish);

	start = system_clock::now();
	soaStart = start;

	__m128 A, B, C, tmpA, tmpB;
	__m128 FOURS = { 4.0f, 4.0f, 4.0f, 4.0f };
	for (int i = 0; i < count; i += 4) {
		A = _mm_loadu_ps(&soaE.a[i]);
		B = _mm_loadu_ps(&soaE.b[i]);
		C = _mm_loadu_ps(&soaE.c[i]);

		tmpA = A;
		tmpB = B;

		A = _mm_mul_ps(A, C);
		B = _mm_mul_ps(B, B);
		A = _mm_mul_ps(A, FOURS);
		B = _mm_sub_ps(B, A);
		B = _mm_sqrt_ps(B);
		B = _mm_sub_ps(B, tmpB);
		tmpA = _mm_add_ps(tmpA, tmpA);
		tmpA = _mm_div_ps(B, tmpA);

		_mm_storeu_ps(&rootsAos[i], tmpA);
	}

	finish = system_clock::now();
	soaFinish = finish;
	printTime("Calculation", start, finish);

	printTime("Aos", aosStart, aosFinish);
	printTime("Soa", soaStart, soaFinish);

	int different = 0;
	for (int i = 0; i < count; i++) {
		float x = roots[i];
		float y = rootsAos[i];
		if (!(std::isnan(x) && std::isnan(y)) && x != y) {
			std::cout << x << ":" << y << "\n";
			different++;
		}
	}
	std::cout << "there were " << different << " differences between the 2 algorithms\n";
	delete[] rootsAos;
	delete[] roots;
	delete[] e;
}
