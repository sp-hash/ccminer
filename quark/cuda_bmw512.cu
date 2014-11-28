#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// die Message it Padding zur Berechnung auf der GPU
__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)
// Init

#if __CUDA_ARCH__ >= 320
__device__
uint64_t bmw_rotl64(const uint64_t x, const int offset)
{
	uint64_t res;
	if (offset<32)
	{
		asm("{\n\t"
			".reg .u32 tl,th,vl;\n\t"
			"mov.b64 {tl,th}, %1;\n\t"
			"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
			"shf.l.wrap.b32 th, th, tl, %2;\n\t"
			"mov.b64 %0, {th,vl};\n\t"
			"}"
			: "=l"(res) : "l"(x), "r"(offset)
			);
	}
	else
	{
		asm("{\n\t"
			".reg .u32 tl,th,vl;\n\t"
			"mov.b64 {tl,th}, %1;\n\t"
			"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
			"shf.l.wrap.b32 th, th, tl, %2;\n\t"
			"mov.b64 %0, {vl,th};\n\t"
			"}"
			: "=l"(res) : "l"(x), "r"(offset)
			);
	}
	return res;
}
#undef ROTL64
#define ROTL64 bmw_rotl64
#endif

#define SHL(x, n)            ((x) << (n))
#define SHR(x, n)            ((x) >> (n))

#define CONST_EXP2(i)    q[i+0] + ROTL64(q[i+1], 5)  + q[i+2] + ROTL64(q[i+3], 11) + \
                    q[i+4] + ROTL64(q[i+5], 27) + q[i+6] + SWAPDWORDS(q[i+7]) + \
                    q[i+8] + ROTL64(q[i+9], 37) + q[i+10] + ROTL64(q[i+11], 43) + \
                    q[i+12] + ROTL64(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])
__device__ __forceinline__ void Compression512_64_first(const uint64_t *const __restrict__ msg, uint64_t *const __restrict__ hash)
{
	
	uint64_t q[32];

	const uint64_t h_c[16] = {
		SPH_C64(0x8081828384858687),
		SPH_C64(0x88898A8B8C8D8E8F),
		SPH_C64(0x9091929394959697),
		SPH_C64(0x98999A9B9C9D9E9F),
		SPH_C64(0xA0A1A2A3A4A5A6A7),
		SPH_C64(0xA8A9AAABACADAEAF),
		SPH_C64(0xB0B1B2B3B4B5B6B7),
		SPH_C64(0xB8B9BABBBCBDBEBF),
		SPH_C64(0xC0C1C2C3C4C5C6C7),
		SPH_C64(0xC8C9CACBCCCDCECF),
		SPH_C64(0xD0D1D2D3D4D5D6D7),
		SPH_C64(0xD8D9DADBDCDDDEDF),
		SPH_C64(0xE0E1E2E3E4E5E6E7),
		SPH_C64(0xE8E9EAEBECEDEEEF),
		SPH_C64(0xF0F1F2F3F4F5F6F7),
		SPH_C64(0xF8F9FAFBFCFDFEFF)
	};


	uint64_t tmp = (msg[5] ^ h_c[5]) - (msg[7] ^ h_c[7]) + (h_c[10]) + (h_c[13]) + (h_c[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + h_c[1];
	uint64_t tmp1 = (msg[6] ^ h_c[6]) - (0x80 ^ h_c[8]) + (h_c[11]) + (h_c[14]) - (512 ^ h_c[15]);
	q[1] = (SHR(tmp1, 1) ^ SHL(tmp1, 2) ^ ROTL64(tmp1, 13) ^ ROTL64(tmp1, 43)) + h_c[2];
	uint64_t tmp2 = (msg[0] ^ h_c[0]) + (msg[7] ^ h_c[7]) + (h_c[9]) - (h_c[12]) + (512 ^ h_c[15]);
	q[2] = (SHR(tmp2, 2) ^ SHL(tmp2, 1) ^ ROTL64(tmp2, 19) ^ ROTL64(tmp2, 53)) + h_c[3];
	uint64_t tmp3 = (msg[0] ^ h_c[0]) - (msg[1] ^ h_c[1]) + (0x80 ^ h_c[8]) - (h_c[10]) + (h_c[13]);
	q[3] = (SHR(tmp3, 2) ^ SHL(tmp3, 2) ^ ROTL64(tmp3, 28) ^ ROTL64(tmp3, 59)) + h_c[4];
	uint64_t tmp4 = (msg[1] ^ h_c[1]) + (msg[2] ^ h_c[2]) + (h_c[9]) - (h_c[11]) - (h_c[14]);
	q[4] = (SHR(tmp4, 1) ^ tmp4) + h_c[5];
	uint64_t tmp5 = (msg[3] ^ h_c[3]) - (msg[2] ^ h_c[2]) + (h_c[10]) - (h_c[12]) + (512 ^ h_c[15]);
	q[5] = (SHR(tmp5, 1) ^ SHL(tmp5, 3) ^ ROTL64(tmp5, 4) ^ ROTL64(tmp5, 37)) + h_c[6];
	uint64_t tmp6 = (msg[4] ^ h_c[4]) - (msg[0] ^ h_c[0]) - (msg[3] ^ h_c[3]) - (h_c[11]) + (h_c[13]);
	q[6] = (SHR(tmp6, 1) ^ SHL(tmp6, 2) ^ ROTL64(tmp6, 13) ^ ROTL64(tmp6, 43)) + h_c[7];
	uint64_t tmp7 = (msg[1] ^ h_c[1]) - (msg[4] ^ h_c[4]) - (msg[5] ^ h_c[5]) - (h_c[12]) - (h_c[14]);
	q[7] = (SHR(tmp7, 2) ^ SHL(tmp7, 1) ^ ROTL64(tmp7, 19) ^ ROTL64(tmp7, 53)) + h_c[8];
	uint64_t tmp8 = (msg[2] ^ h_c[2]) - (msg[5] ^ h_c[5]) - (msg[6] ^ h_c[6]) + (h_c[13]) - (512 ^ h_c[15]);
	q[8] = (SHR(tmp8, 2) ^ SHL(tmp8, 2) ^ ROTL64(tmp8, 28) ^ ROTL64(tmp8, 59)) + h_c[9];
	uint64_t tmp9 = (msg[0] ^ h_c[0]) - (msg[3] ^ h_c[3]) + (msg[6] ^ h_c[6]) - (msg[7] ^ h_c[7]) + (h_c[14]);
	q[9] = (SHR(tmp9, 1) ^ tmp9) + h_c[10];
	uint64_t tmpa = (0x80 ^ h_c[8]) - (msg[1] ^ h_c[1]) - (msg[4] ^ h_c[4]) - (msg[7] ^ h_c[7]) + (512 ^ h_c[15]);
	q[10] = (SHR(tmpa, 1) ^ SHL(tmpa, 3) ^ ROTL64(tmpa, 4) ^ ROTL64(tmpa, 37)) + h_c[11];
	uint64_t tmpb = (0x80 ^ h_c[8]) - (msg[0] ^ h_c[0]) - (msg[2] ^ h_c[2]) - (msg[5] ^ h_c[5]) + (h_c[9]);
	q[11] = (SHR(tmpb, 1) ^ SHL(tmpb, 2) ^ ROTL64(tmpb, 13) ^ ROTL64(tmpb, 43)) + h_c[12];
	uint64_t tmpc = (msg[1] ^ h_c[1]) + (msg[3] ^ h_c[3]) - (msg[6] ^ h_c[6]) - (h_c[9]) + (h_c[10]);
	q[12] = (SHR(tmpc, 2) ^ SHL(tmpc, 1) ^ ROTL64(tmpc, 19) ^ ROTL64(tmpc, 53)) + h_c[13];
	uint64_t tmpd = (msg[2] ^ h_c[2]) + (msg[4] ^ h_c[4]) + (msg[7] ^ h_c[7]) + (h_c[10]) + (h_c[11]);
	q[13] = (SHR(tmpd, 2) ^ SHL(tmpd, 2) ^ ROTL64(tmpd, 28) ^ ROTL64(tmpd, 59)) + h_c[14];
	uint64_t tmpe = (msg[3] ^ h_c[3]) - (msg[5] ^ h_c[5]) + (0x80 ^ h_c[8]) - (h_c[11]) - (h_c[12]);
	q[14] = (SHR(tmpe, 1) ^ tmpe) + h_c[15];
	uint64_t tmpf = (h_c[12]) - (msg[4] ^ h_c[4]) - (msg[6] ^ h_c[6]) - (h_c[9]) + (h_c[13]);
	q[15] = (SHR(tmpf, 1) ^ SHL(tmpf, 3) ^ ROTL64(tmpf, 4) ^ ROTL64(tmpf, 37)) + h_c[0];

	q[0 + 16] =
		(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROTL64(q[0], 13) ^ ROTL64(q[0], 43)) +
		(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROTL64(q[0 + 1], 19) ^ ROTL64(q[0 + 1], 53)) +
		(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROTL64(q[0 + 2], 28) ^ ROTL64(q[0 + 2], 59)) +
		(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROTL64(q[0 + 3], 4) ^ ROTL64(q[0 + 3], 37)) +
		(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROTL64(q[0 + 4], 13) ^ ROTL64(q[0 + 4], 43)) +
		(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROTL64(q[0 + 5], 19) ^ ROTL64(q[0 + 5], 53)) +
		(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROTL64(q[0 + 6], 28) ^ ROTL64(q[0 + 6], 59)) +
		(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROTL64(q[0 + 7], 4) ^ ROTL64(q[0 + 7], 37)) +
		(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROTL64(q[0 + 8], 13) ^ ROTL64(q[0 + 8], 43)) +
		(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROTL64(q[0 + 9], 19) ^ ROTL64(q[0 + 9], 53)) +
		(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROTL64(q[0 + 10], 28) ^ ROTL64(q[0 + 10], 59)) +
		(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROTL64(q[0 + 11], 4) ^ ROTL64(q[0 + 11], 37)) +
		(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROTL64(q[0 + 12], 13) ^ ROTL64(q[0 + 12], 43)) +
		(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROTL64(q[0 + 13], 19) ^ ROTL64(q[0 + 13], 53)) +
		(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROTL64(q[0 + 14], 28) ^ ROTL64(q[0 + 14], 59)) +
		(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROTL64(q[0 + 15], 4) ^ ROTL64(q[0 + 15], 37)) +
		((((0 + 16)*(0x0555555555555555ull)) + ROTL64(msg[0], 0 + 1) +
		ROTL64(msg[0 + 3], 0 + 4) ) ^ h_c[0 + 7]);
	q[1 + 16] =
		(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROTL64(q[1], 13) ^ ROTL64(q[1], 43)) +
		(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROTL64(q[1 + 1], 19) ^ ROTL64(q[1 + 1], 53)) +
		(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROTL64(q[1 + 2], 28) ^ ROTL64(q[1 + 2], 59)) +
		(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROTL64(q[1 + 3], 4) ^ ROTL64(q[1 + 3], 37)) +
		(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROTL64(q[1 + 4], 13) ^ ROTL64(q[1 + 4], 43)) +
		(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROTL64(q[1 + 5], 19) ^ ROTL64(q[1 + 5], 53)) +
		(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROTL64(q[1 + 6], 28) ^ ROTL64(q[1 + 6], 59)) +
		(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROTL64(q[1 + 7], 4) ^ ROTL64(q[1 + 7], 37)) +
		(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROTL64(q[1 + 8], 13) ^ ROTL64(q[1 + 8], 43)) +
		(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROTL64(q[1 + 9], 19) ^ ROTL64(q[1 + 9], 53)) +
		(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROTL64(q[1 + 10], 28) ^ ROTL64(q[1 + 10], 59)) +
		(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROTL64(q[1 + 11], 4) ^ ROTL64(q[1 + 11], 37)) +
		(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROTL64(q[1 + 12], 13) ^ ROTL64(q[1 + 12], 43)) +
		(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROTL64(q[1 + 13], 19) ^ ROTL64(q[1 + 13], 53)) +
		(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROTL64(q[1 + 14], 28) ^ ROTL64(q[1 + 14], 59)) +
		(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROTL64(q[1 + 15], 4) ^ ROTL64(q[1 + 15], 37)) +
		((((1 + 16)*(0x0555555555555555ull)) + ROTL64(msg[1], 1 + 1) +
		ROTL64(msg[1 + 3], 1 + 4) ) ^ h_c[1 + 7]);

	q[2 + 16] = q[2 + 0] + ROTL64(q[2 + 1], 5) + q[2 + 2] + ROTL64(q[2 + 3], 11) + \
		q[2 + 4] + ROTL64(q[2 + 5], 27) + q[2 + 6] + SWAPDWORDS(q[2 + 7]) + \
		q[2 + 8] + ROTL64(q[2 + 9], 37) + q[2 + 10] + ROTL64(q[2 + 11], 43) + \
		q[2 + 12] + ROTL64(q[2 + 13], 53) + (SHR(q[2 + 14], 1) ^ q[2 + 14]) + (SHR(q[2 + 15], 2) ^ q[2 + 15])		
		+ ((((2 + 16)*(0x0555555555555555ull)) + ROTL64(msg[2], 2 + 1) +
		ROTL64(msg[2 + 3], 2 + 4) - ROTL64(msg[2 + 10], 2 + 11)) ^ h_c[2 + 7]);
	q[3 + 16] = q[3 + 0] + ROTL64(q[3 + 1], 5) + q[3 + 2] + ROTL64(q[3 + 3], 11) + \
		q[3 + 4] + ROTL64(q[3 + 5], 27) + q[3 + 6] + SWAPDWORDS(q[3 + 7]) + \
		q[3 + 8] + ROTL64(q[3 + 9], 37) + q[3 + 10] + ROTL64(q[3 + 11], 43) + \
		q[3 + 12] + ROTL64(q[3 + 13], 53) + (SHR(q[3 + 14], 1) ^ q[3 + 14]) + (SHR(q[3 + 15], 2) ^ q[3 + 15])
		+((((3 + 16)*(0x0555555555555555ull)) + ROTL64(msg[3], 3 + 1) +
		ROTL64(msg[3 + 3], 3 + 4) - ROTL64(msg[3 + 10], 3 + 11)) ^ h_c[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((((4 + 16)*(0x0555555555555555ull)) + ROTL64(msg[4], 4 + 1) +
		ROTL64(msg[4 + 3], 4 + 4) ) ^ h_c[4 + 7]);


	q[21] = q[5] + ROTL64(q[6], 5) + q[7] + ROTL64(q[8], 11) + 
		q[9] + ROTL64(q[10], 27) + q[11] + SWAPDWORDS(q[12]) + 
		q[13] + ROTL64(q[14], 37) + q[15] + ROTL64(q[16], 43) + 
		q[17] + ROTL64(q[18], 53) + (SHR(q[19], 1) ^ q[19]) + (SHR(q[20], 2) ^ q[20]) +
		((((21)*(0x0555555555555555ull)) + ROTL64(msg[5], 6) +
		0x10000 - 0x2000000) ^ h_c[12]);

	q[6 + 16] = CONST_EXP2(6) +
		((((6 + 16)*(0x0555555555555555ull)) + ROTL64(msg[6], 6 + 1) -
		ROTL64(msg[6 - 6], (6 - 6) + 1)) ^ h_c[6 + 7]);
	q[7 + 16] = CONST_EXP2(7) +
		((((7 + 16)*(0x0555555555555555ull)) + ROTL64(msg[7], 7 + 1) -
		ROTL64(msg[7 - 6], (7 - 6) + 1)) ^ h_c[7 + 7]);
	q[8 + 16] = CONST_EXP2(8) +
		((((8 + 16)*(0x0555555555555555ull)) + 0x0000000000010000ULL -
		ROTL64(msg[8 - 6], (8 - 6) + 1)) ^ h_c[8 + 7]);

	q[9 + 16] = CONST_EXP2(9) +
		((((9 + 16)*(0x0555555555555555ull)) -
		ROTL64(msg[9 - 6], (9 - 6) + 1)) ^ h_c[9 - 9]);
	q[10 + 16] = CONST_EXP2(10) +
		((((10 + 16)*(0x0555555555555555ull)) -
		ROTL64(msg[10 - 6], (10 - 6) + 1)) ^ h_c[10 - 9]);
	q[11 + 16] = CONST_EXP2(11) +
		((((11 + 16)*(0x0555555555555555ull)) -
		ROTL64(msg[11 - 6], (11 - 6) + 1)) ^ h_c[11 - 9]);
	q[12 + 16] = CONST_EXP2(12) +
		((((12 + 16)*(0x0555555555555555ull)) +
		0x0000000002000000ull - ROTL64(msg[12 - 6], (12 - 6) + 1)) ^ h_c[12 - 9]);

	q[13 + 16] = CONST_EXP2(13) +
		((((13 + 16)*(0x0555555555555555ull)) +
		ROTL64(msg[13 - 13], (13 - 13) + 1) - ROTL64(msg[13 - 6], (13 - 6) + 1)) ^ h_c[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((((14 + 16)*(0x0555555555555555ull)) +
		ROTL64(msg[14 - 13], (14 - 13) + 1) - 0x0000000000010000ull) ^ h_c[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((((15 + 16)*(0x0555555555555555ull)) + ROTL64(512ULL, 15 + 1) +
		ROTL64(msg[15 - 13], (15 - 13) + 1)) ^ h_c[15 - 9]);

	uint64_t XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	uint64_t XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ msg[0]) + (XL64    ^ q[24] ^ q[0]);
	hash[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ msg[1]) + (XL64    ^ q[25] ^ q[1]);
	hash[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ msg[2]) + (XL64    ^ q[26] ^ q[2]);
	hash[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ msg[3]) + (XL64    ^ q[27] ^ q[3]);
	hash[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + (XL64    ^ q[28] ^ q[4]);
	hash[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ msg[5]) + (XL64    ^ q[29] ^ q[5]);
	hash[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ msg[6]) + (XL64    ^ q[30] ^ q[6]);
	hash[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ msg[7]) + (XL64    ^ q[31] ^ q[7]);

	hash[8] = ROTL64(hash[4], 9) + (XH64     ^     q[24] ^ 0x0000000000000080U) + (SHL(XL64, 8) ^ q[23] ^ q[8]);
	hash[9] = ROTL64(hash[5], 10) + (XH64     ^     q[25]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
	hash[10] = ROTL64(hash[6], 11) + (XH64     ^     q[26]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
	hash[11] = ROTL64(hash[7], 12) + (XH64     ^     q[27]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
	hash[12] = ROTL64(hash[0], 13) + (XH64     ^     q[28]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
	hash[13] = ROTL64(hash[1], 14) + (XH64     ^     q[29]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
	hash[14] = ROTL64(hash[2], 15) + (XH64     ^     q[30]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
	hash[15] = ROTL16(hash[3]) + (XH64     ^     q[31] ^ 512) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}

__device__ __forceinline__ void Compression512(uint64_t *msg, uint64_t *hash)
{
    // Compression ref. implementation
    uint64_t tmp;
    uint64_t q[32];

    tmp = (msg[ 5] ^ hash[ 5]) - (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
    q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[1];
    tmp = (msg[ 6] ^ hash[ 6]) - (msg[ 8] ^ hash[ 8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
    q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[2];
    tmp = (msg[ 0] ^ hash[ 0]) + (msg[ 7] ^ hash[ 7]) + (msg[ 9] ^ hash[ 9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
    q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[3];
    tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 1] ^ hash[ 1]) + (msg[ 8] ^ hash[ 8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
    q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[4];
    tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 2] ^ hash[ 2]) + (msg[ 9] ^ hash[ 9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
    q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
    tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 2] ^ hash[ 2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
    q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[6];
    tmp = (msg[ 4] ^ hash[ 4]) - (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
    q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[7];
    tmp = (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 5] ^ hash[ 5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
    q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[8];
    tmp = (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) - (msg[ 6] ^ hash[ 6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
    q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[9];
    tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) + (msg[ 6] ^ hash[ 6]) - (msg[ 7] ^ hash[ 7]) + (msg[14] ^ hash[14]);
    q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
    tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 7] ^ hash[ 7]) + (msg[15] ^ hash[15]);
    q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp,  4) ^ ROTL64(tmp, 37)) + hash[11];
    tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 0] ^ hash[ 0]) - (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) + (msg[ 9] ^ hash[ 9]);
    q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROTL64(tmp, 13) ^ ROTL64(tmp, 43)) + hash[12];
    tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 3] ^ hash[ 3]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[10] ^ hash[10]);
    q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROTL64(tmp, 19) ^ ROTL64(tmp, 53)) + hash[13];
    tmp = (msg[ 2] ^ hash[ 2]) + (msg[ 4] ^ hash[ 4]) + (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
    q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROTL64(tmp, 28) ^ ROTL64(tmp, 59)) + hash[14];
    tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 5] ^ hash[ 5]) + (msg[ 8] ^ hash[ 8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
    q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
    tmp = (msg[12] ^ hash[12]) - (msg[ 4] ^ hash[ 4]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[13] ^ hash[13]);
    q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROTL64(tmp, 4) ^ ROTL64(tmp, 37)) + hash[0];
	q[0 + 16] =
		(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROTL64(q[0], 13) ^ ROTL64(q[0], 43)) +
		(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROTL64(q[0 + 1], 19) ^ ROTL64(q[0 + 1], 53)) +
		(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROTL64(q[0 + 2], 28) ^ ROTL64(q[0 + 2], 59)) +
		(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROTL64(q[0 + 3], 4) ^ ROTL64(q[0 + 3], 37)) +
		(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROTL64(q[0 + 4], 13) ^ ROTL64(q[0 + 4], 43)) +
		(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROTL64(q[0 + 5], 19) ^ ROTL64(q[0 + 5], 53)) +
		(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROTL64(q[0 + 6], 28) ^ ROTL64(q[0 + 6], 59)) +
		(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROTL64(q[0 + 7], 4) ^ ROTL64(q[0 + 7], 37)) +
		(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROTL64(q[0 + 8], 13) ^ ROTL64(q[0 + 8], 43)) +
		(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROTL64(q[0 + 9], 19) ^ ROTL64(q[0 + 9], 53)) +
		(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROTL64(q[0 + 10], 28) ^ ROTL64(q[0 + 10], 59)) +
		(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROTL64(q[0 + 11], 4) ^ ROTL64(q[0 + 11], 37)) +
		(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROTL64(q[0 + 12], 13) ^ ROTL64(q[0 + 12], 43)) +
		(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROTL64(q[0 + 13], 19) ^ ROTL64(q[0 + 13], 53)) +
		(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROTL64(q[0 + 14], 28) ^ ROTL64(q[0 + 14], 59)) +
		(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROTL64(q[0 + 15], 4) ^ ROTL64(q[0 + 15], 37)) +
		((((0 + 16)*(0x0555555555555555ull)) + ROTL64(msg[0], 0 + 1) +
		ROTL64(msg[0 + 3], 0 + 4) - ROTL64(msg[0 + 10], 0 + 11)) ^ hash[0 + 7]);
	q[1 + 16] =
		(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROTL64(q[1], 13) ^ ROTL64(q[1], 43)) +
		(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROTL64(q[1 + 1], 19) ^ ROTL64(q[1 + 1], 53)) +
		(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROTL64(q[1 + 2], 28) ^ ROTL64(q[1 + 2], 59)) +
		(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROTL64(q[1 + 3], 4) ^ ROTL64(q[1 + 3], 37)) +
		(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROTL64(q[1 + 4], 13) ^ ROTL64(q[1 + 4], 43)) +
		(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROTL64(q[1 + 5], 19) ^ ROTL64(q[1 + 5], 53)) +
		(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROTL64(q[1 + 6], 28) ^ ROTL64(q[1 + 6], 59)) +
		(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROTL64(q[1 + 7], 4) ^ ROTL64(q[1 + 7], 37)) +
		(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROTL64(q[1 + 8], 13) ^ ROTL64(q[1 + 8], 43)) +
		(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROTL64(q[1 + 9], 19) ^ ROTL64(q[1 + 9], 53)) +
		(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROTL64(q[1 + 10], 28) ^ ROTL64(q[1 + 10], 59)) +
		(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROTL64(q[1 + 11], 4) ^ ROTL64(q[1 + 11], 37)) +
		(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROTL64(q[1 + 12], 13) ^ ROTL64(q[1 + 12], 43)) +
		(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROTL64(q[1 + 13], 19) ^ ROTL64(q[1 + 13], 53)) +
		(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROTL64(q[1 + 14], 28) ^ ROTL64(q[1 + 14], 59)) +
		(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROTL64(q[1 + 15], 4) ^ ROTL64(q[1 + 15], 37)) +
		((((1 + 16)*(0x0555555555555555ull)) + ROTL64(msg[1], 1 + 1) +
		ROTL64(msg[1 + 3], 1 + 4) - ROTL64(msg[1 + 10], 1 + 11)) ^ hash[1 + 7]);


	q[2 + 16] = CONST_EXP2(2) +
		((((2 + 16)*(0x0555555555555555ull)) + ROTL64(msg[2], 2 + 1) +
		ROTL64(msg[2 + 3], 2 + 4) - ROTL64(msg[2 + 10], 2 + 11)) ^ hash[2 + 7]);
	q[3 + 16] = CONST_EXP2(3) +
		((((3 + 16)*(0x0555555555555555ull)) + ROTL64(msg[3], 3 + 1) +
		ROTL64(msg[3 + 3], 3 + 4) - ROTL64(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((((4 + 16)*(0x0555555555555555ull)) + ROTL64(msg[4], 4 + 1) +
		ROTL64(msg[4 + 3], 4 + 4) - ROTL64(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	q[5 + 16] = CONST_EXP2(5) +
		((((5 + 16)*(0x0555555555555555ull)) + ROTL64(msg[5], 5 + 1) +
		ROTL64(msg[5 + 3], 5 + 4) - ROTL16(msg[5 + 10])) ^ hash[5 + 7]);

	q[6 + 16] = CONST_EXP2(6) +
		((((6 + 16)*(0x0555555555555555ull)) + ROTL64(msg[6], 6 + 1) +
		ROTL64(msg[6 + 3], 6 + 4) - ROTL64(msg[6 - 6], (6 - 6) + 1)) ^ hash[6 + 7]);
	q[7 + 16] = CONST_EXP2(7) +
		((((7 + 16)*(0x0555555555555555ull)) + ROTL64(msg[7], 7 + 1) +
		ROTL64(msg[7 + 3], 7 + 4) - ROTL64(msg[7 - 6], (7 - 6) + 1)) ^ hash[7 + 7]);
	q[8 + 16] = CONST_EXP2(8) +
		((((8 + 16)*(0x0555555555555555ull)) + ROTL64(msg[8], 8 + 1) +
		ROTL64(msg[8 + 3], 8 + 4) - ROTL64(msg[8 - 6], (8 - 6) + 1)) ^ hash[8 + 7]);

	q[9 + 16] = CONST_EXP2(9) +
		((((9 + 16)*(0x0555555555555555ull)) + ROTL64(msg[9], 9 + 1) +
		ROTL64(msg[9 + 3], 9 + 4) - ROTL64(msg[9 - 6], (9 - 6) + 1)) ^ hash[9 - 9]);
	q[10 + 16] = CONST_EXP2(10) +
		((((10 + 16)*(0x0555555555555555ull)) + ROTL64(msg[10], 10 + 1) +
		ROTL64(msg[10 + 3], 10 + 4) - ROTL64(msg[10 - 6], (10 - 6) + 1)) ^ hash[10 - 9]);
	q[11 + 16] = CONST_EXP2(11) +
		((((11 + 16)*(0x0555555555555555ull)) + ROTL64(msg[11], 11 + 1) +
		ROTL64(msg[11 + 3], 11 + 4) - ROTL64(msg[11 - 6], (11 - 6) + 1)) ^ hash[11 - 9]);
	q[12 + 16] = CONST_EXP2(12) +
		((((12 + 16)*(0x0555555555555555ull)) + ROTL64(msg[12], 12 + 1) +
		ROTL64(msg[12 + 3], 12 + 4) - ROTL64(msg[12 - 6], (12 - 6) + 1)) ^ hash[12 - 9]);

	q[13 + 16] = CONST_EXP2(13) +
		((((13 + 16)*(0x0555555555555555ull)) + ROTL64(msg[13], 13 + 1) +
		ROTL64(msg[13 - 13], (13 - 13) + 1) - ROTL64(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((((14 + 16)*(0x0555555555555555ull)) + ROTL64(msg[14], 14 + 1) +
		ROTL64(msg[14 - 13], (14 - 13) + 1) - ROTL64(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((((15 + 16)*(0x0555555555555555ull)) + ROTL64(msg[15], 15 + 1) +
		ROTL64(msg[15 - 13], (15 - 13) + 1) - ROTL64(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);

    uint64_t XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
    uint64_t XH64 = XL64^q[24]^q[25]^q[26]^q[27]^q[28]^q[29]^q[30]^q[31];

    hash[0] =                       (SHL(XH64, 5) ^ SHR(q[16],5) ^ msg[ 0]) + (    XL64    ^ q[24] ^ q[ 0]);
    hash[1] =                       (SHR(XH64, 7) ^ SHL(q[17],8) ^ msg[ 1]) + (    XL64    ^ q[25] ^ q[ 1]);
    hash[2] =                       (SHR(XH64, 5) ^ SHL(q[18],5) ^ msg[ 2]) + (    XL64    ^ q[26] ^ q[ 2]);
    hash[3] =                       (SHR(XH64, 1) ^ SHL(q[19],5) ^ msg[ 3]) + (    XL64    ^ q[27] ^ q[ 3]);
    hash[4] =                       (SHR(XH64, 3) ^     q[20]    ^ msg[ 4]) + (    XL64    ^ q[28] ^ q[ 4]);
    hash[5] =                       (SHL(XH64, 6) ^ SHR(q[21],6) ^ msg[ 5]) + (    XL64    ^ q[29] ^ q[ 5]);
    hash[6] =                       (SHR(XH64, 4) ^ SHL(q[22],6) ^ msg[ 6]) + (    XL64    ^ q[30] ^ q[ 6]);
    hash[7] =                       (SHR(XH64,11) ^ SHL(q[23],2) ^ msg[ 7]) + (    XL64    ^ q[31] ^ q[ 7]);

    hash[ 8] = ROTL64(hash[4], 9) + (    XH64     ^     q[24]    ^ msg[ 8]) + (SHL(XL64,8) ^ q[23] ^ q[ 8]);
    hash[ 9] = ROTL64(hash[5],10) + (    XH64     ^     q[25]    ^ msg[ 9]) + (SHR(XL64,6) ^ q[16] ^ q[ 9]);
    hash[10] = ROTL64(hash[6],11) + (    XH64     ^     q[26]    ^ msg[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
    hash[11] = ROTL64(hash[7],12) + (    XH64     ^     q[27]    ^ msg[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
    hash[12] = ROTL64(hash[0],13) + (    XH64     ^     q[28]    ^ msg[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
    hash[13] = ROTL64(hash[1],14) + (    XH64     ^     q[29]    ^ msg[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
    hash[14] = ROTL64(hash[2],15) + (    XH64     ^     q[30]    ^ msg[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
    hash[15] = ROTL16(hash[3]) + (    XH64     ^     q[31]    ^ msg[15]) + (SHR(XL64,2) ^ q[22] ^ q[15]);
}

__global__ __launch_bounds__(256, 2)
void quark_bmw512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint64_t *inpHash = &g_hash[8 * hashPosition];

        // Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
        // BMW arbeitet mit 128 Byte!!!
		uint64_t h[16];
		uint64_t message[16];
#pragma unroll 8
        for(int i=0;i<8;i++)
            message[i] = inpHash[i];

        // Compression 1
        Compression512_64_first(message, h);

        // Final
#pragma unroll 16
        for(int i=0;i<16;i++)
            message[i] = 0xaaaaaaaaaaaaaaa0ull + (uint64_t)i;

        Compression512(h, message);

        // fertig
        uint64_t *outpHash = &g_hash[8 * hashPosition];

#pragma unroll 8
        for(int i=0;i<8;i++)
            outpHash[i] = message[i+8];
    }
}

__global__ __launch_bounds__(256,2)
void quark_bmw512_gpu_hash_80(int threads, uint32_t startNounce, uint64_t *g_hash)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = startNounce + thread;

        // Init
		uint64_t h[16];
        // Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
        // BMW arbeitet mit 128 Byte!!!
        uint64_t message[16];
#pragma unroll 16
        for(int i=0;i<16;i++)
            message[i] = c_PaddedMessage80[i];

        // die Nounce durch die thread-spezifische ersetzen
        message[9] = REPLACE_HIWORD(message[9], cuda_swab32(nounce));

        // Compression 1
        Compression512(message, h);

        // Final
#pragma unroll 16
        for(int i=0;i<16;i++)
            message[i] = 0xaaaaaaaaaaaaaaa0ull + (uint64_t)i;

        Compression512(h, message);

        // fertig
        uint64_t *outpHash = &g_hash[8 * thread];

#pragma unroll 8
        for(int i=0;i<8;i++)
            outpHash[i] = message[i+8];
    }
}

// Setup-Funktionen
__host__ void quark_bmw512_cpu_init(int thr_id, int threads)
{
    // nix zu tun ;-)
	// jetzt schon :D
}

// Bmw512 für 80 Byte grosse Eingangsdaten
__host__ void quark_bmw512_cpu_setBlock_80(void *pdata)
{
	// Message mit Padding bereitstellen
	// lediglich die korrekte Nonce ist noch ab Byte 76 einzusetzen.
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	uint64_t *message = (uint64_t*)PaddedMessage;
	// Padding einfügen (Byteorder?!?)
	message[10] = SPH_C64(0x80);
	// Länge (in Bits, d.h. 80 Byte * 8 = 640 Bits
	message[15] = SPH_C64(640);

	// die Message zur Berechnung auf der GPU
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__ void quark_bmw512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const int threadsperblock = 64;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_bmw512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
    MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void quark_bmw512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order)
{
	const int threadsperblock = 64;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_bmw512_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash);
    MyStreamSynchronize(NULL, order, thr_id);
}

