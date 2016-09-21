/**
 * The function to optimise as part of the coursework.
 *
 * l0, l1, l2 and l3 record the amount of time spent in each loop
 * and should not be optimised out. :)
 */

#include <immintrin.h>
#include <omp.h>
void compute() {

	omp_set_num_threads(4);
	int S = 4;
	int vec_size = (N/4)*4;

	double t0, t1;

	// Loop 0.
	//optimistions to loop 0: loop fission to allow contiguous blocks in cache, and then unroll to allow stride through the cache
	t0 = wtime();

{
	__m128 ini = _mm_set1_ps(0.0f);

	#pragma unroll(16);
	for (int i = 0; i < vec_size; i+= S) {
		_mm_store_ps(ax+i, ini);
	}

	for (int i = vec_size; i<N;i++){
		ax[i] = 0.0f;
	}

	#pragma unroll(16);
	for (int i = 0; i < vec_size; i+= S) {
		_mm_store_ps(ay+i, ini);
	}

	for (int i = vec_size; i<N;i++){
		ay[i] = 0.0f;
	}

	#pragma unroll(16);
	for (int i = 0; i < vec_size; i+=S) {
		_mm_store_ps(az+i, ini);
	}
	for (int i = vec_size; i<N;i++){
		az[i] = 0.0f;
	}

	t1 = wtime();
	l0 += (t1 - t0);
}
    // Loop 1.
	t0 = wtime();
{

	/*for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float rx = x[j] - x[i];
			float ry = y[j] - y[i];
			float rz = z[j] - z[i];
			float r2 = rx*rx + ry*ry + rz*rz + eps;
			float r2inv = 1.0f / sqrt(r2);
			float r6inv = r2inv * r2inv * r2inv;
			float s = m[j] * r6inv;
			ax[i] += s * rx;
			ay[i] += s * ry;
			az[i] += s * rz;
			printf("%d\n", ax[i] );
		}
	}*/

	//pragma to parallelise the code.
	#pragma omp parallel for schedule(static) shared(N)
	for (int i = 0; i < vec_size; i+=4) {
		//registers used to hold current xyz values
		__m128 xi_v = _mm_load_ps(x+i);
		__m128 yi_v = _mm_load_ps(y+i);
		__m128 zi_v = _mm_load_ps(z+i);
		float l;
		float ms;
		//registers to accumulate and store the results
		__m128 ax_v = _mm_load_ps(ax+i);
		__m128 ay_v = _mm_load_ps(ay+i);
		__m128 az_v = _mm_load_ps(az+i);
		__m128 accx = _mm_set1_ps(0.0f);
		__m128 accy = _mm_set1_ps(0.0f);
		__m128 accz = _mm_set1_ps(0.0f);

		__m128 eps_v = _mm_set1_ps(eps);

		#pragma unroll(16)
		for (int j = 0; j < N; j++) {
			// /float rx = x[j] - x[i];
			l = x[j];
			__m128 vj_v = _mm_set1_ps(l);
			__m128 rx = _mm_sub_ps(vj_v, xi_v);
			__m128 r2 = _mm_mul_ps(rx, rx);

			//float ry = y[j] - y[i];
			l = y[j];
			vj_v = _mm_set1_ps(l);
			__m128 ry = _mm_sub_ps(vj_v, yi_v);
			r2 = _mm_add_ps(r2, _mm_mul_ps(ry, ry));

			//float rz = z[j] - z[i];
			l = z[j];

			vj_v = _mm_set1_ps(l);
			__m128 rz = _mm_sub_ps(vj_v, zi_v);
			r2 = _mm_add_ps(r2, _mm_mul_ps(rz, rz));

			//float r2 = rx*rx + ry*ry + rz*rz + eps;
			r2 = _mm_add_ps(r2, eps_v);

			//float r2inv = 1.0f / sqrt(r2);

			r2 = _mm_rsqrt_ps(r2);
			

			//float r6inv = r2inv * r2inv * r2inv;
		 	r2 = _mm_mul_ps(_mm_mul_ps(r2, r2), r2);

			//float s = m[j] * r6inv;
			ms = m[j];
			__m128 m_v = _mm_set1_ps(ms);
			__m128 s_v = _mm_mul_ps(m_v, r2);


			//ax[i] += s * rx;
			accx = _mm_add_ps(accx, _mm_mul_ps(s_v, rx));

			//ay[i] += s * ry;
			accy = _mm_add_ps(accy, _mm_mul_ps(s_v, ry));

			//az[i] += s * rz;
			accz = _mm_add_ps(accz, _mm_mul_ps(s_v, rz));

			//printf("%d\n", ax[i]);
		}
		_mm_store_ps(ax+i,_mm_add_ps(ax_v, accx));
		_mm_store_ps(ay+i,_mm_add_ps(ax_v, accy));
		_mm_store_ps(az+i,_mm_add_ps(ax_v, accz));
	}

	//cleanup code. This is needed for inputs which are not a multiple of 4, else the last few items wont be processed.
	for (int i = vec_size; i < N; i++) {
		#pragma unroll(16)
		for (int j = 0; j < N; j++) {
			float rx = x[j] - x[i];
			float ry = y[j] - y[i];
			float rz = z[j] - z[i];
			float r2 = rx*rx + ry*ry + rz*rz + eps;
			float r2inv = 1.0f / sqrt(r2);
			float r6inv = r2inv * r2inv * r2inv;
			float s = m[j] * r6inv;
			ax[i] += s * rx;
			ay[i] += s * ry;
			az[i] += s * rz;
		}
	}
	t1 = wtime();
	l1 += (t1 - t0);
}
	// Loop 2.
	t0 = wtime();

{

	__m128 dt_v = _mm_set1_ps(dt);
	__m128 dmp_v = _mm_set1_ps(dmp);
	#pragma unroll(16)
	for (int i = 0; i < vec_size; i+=S) {
		//vx[i] += dmp * (dt * ax[i]);

		__m128 ax_v = _mm_load_ps(ax+i);
		__m128 vx_v = _mm_load_ps(vx+i);
		__m128 r1 = _mm_mul_ps(dt_v, ax_v);
		__m128 r2 = _mm_mul_ps(dmp_v, r1);
		__m128 r3 = _mm_add_ps(vx_v, r2);

		_mm_store_ps(vx+i, r3);

	}
	for (int i = vec_size;i<N;i++){
		vx[i] += dmp * (dt * ax[i]);
	}

	#pragma unroll(16)
	for (int i = 0; i < vec_size; i+=4) {
		__m128 ay_v = _mm_load_ps(ay+i);
		__m128 vy_v = _mm_load_ps(vy+i);

		__m128 r1 = _mm_mul_ps(dt_v, ay_v);
		__m128 r2 = _mm_mul_ps(dmp_v, r1);
		__m128 r3 = _mm_add_ps(vy_v, r2);

		_mm_store_ps(vy+i, r3);

	}
	for (int i = vec_size;i<N;i++){
		vy[i] += dmp * (dt * ay[i]);
	}
	#pragma unroll(16)
	for (int i = 0; i < vec_size; i+=4) {
		__m128 az_v = _mm_load_ps(az+i);
		__m128 vz_v = _mm_load_ps(vz+i);

		__m128 r1 = _mm_mul_ps(dt_v, az_v);
		__m128 r2 = _mm_mul_ps(dmp_v, r1);
		__m128 r3 = _mm_add_ps(vz_v, r2);

		_mm_store_ps(vz+i, r3);
	}
	for (int i = vec_size;i<N;i++){
		vz[i] += dmp * (dt * az[i]);
	}

	t1 = wtime();
	l2 += (t1 - t0);
}
	// Loop 3.
	t0 = wtime();
{
	__m128 mone = _mm_set1_ps(-1.0f);
	__m128 one = _mm_set1_ps(1.0f);
	__m128 dt_v = _mm_set1_ps(dt);

	#pragma unroll(16);
	for (int i = 0; i < vec_size; i+=4) {
		//x[i] += dt * vx[i];
		__m128 x_v = _mm_load_ps(x+i);
		__m128 vx_v = _mm_load_ps(vx+i);
		__m128 r1 = _mm_add_ps(_mm_mul_ps(vx_v, dt_v), x_v);
		_mm_store_ps(x+i, r1);

		//if(x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
		__m128 mul = _mm_mul_ps(vx_v, mone);
		__m128 maska = _mm_cmple_ps(r1, mone);
		__m128 maskb = _mm_cmpge_ps(r1, one);
		__m128 mask = _mm_or_ps(maska, maskb);
		__m128 res = _mm_or_ps(_mm_and_ps(mask, mul), _mm_andnot_ps(mask, vx_v));
		_mm_store_ps(vx+i, res );

	}
	#pragma unroll(16);
	for (int i = 0; i < vec_size; i+=4) {
		//x[i] += dt * vx[i];
		__m128 y_v = _mm_load_ps(y+i);
		__m128 vy_v = _mm_load_ps(vy+i);
		__m128 r1 = _mm_add_ps(_mm_mul_ps(vy_v, dt_v), y_v);
		_mm_store_ps(y+i, r1);

		//if(x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
		__m128 mul = _mm_mul_ps(vy_v, mone);
		__m128 maska = _mm_cmple_ps(r1, mone);
		__m128 maskb = _mm_cmpge_ps(r1, one);
		__m128 mask = _mm_or_ps(maska, maskb);
		__m128 res = _mm_or_ps(_mm_and_ps(mask, mul), _mm_andnot_ps(mask, vy_v));
		_mm_store_ps(vy+i, res );

	}
	#pragma unroll(16);
	for (int i = 0; i < vec_size; i+=4) {
		//x[i] += dt * vx[i];
		__m128 z_v = _mm_load_ps(z+i);
		__m128 vz_v = _mm_load_ps(vz+i);
		__m128 r1 = _mm_add_ps(_mm_mul_ps(vz_v, dt_v), z_v);
		_mm_store_ps(z+i, r1);

		//if(x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
		__m128 mul = _mm_mul_ps(vz_v, mone);
		__m128 maska = _mm_cmple_ps(r1, mone);
		__m128 maskb = _mm_cmpge_ps(r1, one);
		__m128 mask = _mm_or_ps(maska, maskb);
		__m128 res = _mm_or_ps(_mm_and_ps(mask, mul), _mm_andnot_ps(mask, vz_v));
		_mm_store_ps(vz+i, res );

	}
}

	//cleanup code. This is needed for inputs which are not a multiple of 4, else the last few items wont be processed.
	for (int i = vec_size; i < N; i++){
		x[i] += dt * vx[i];
		y[i] += dt * vy[i];
		z[i] += dt * vz[i];
		if(x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
		if(y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
		if(z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
	}
	t1 = wtime();
	l3 += (t1 - t0);

}
