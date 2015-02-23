#include <string.h>
#include <stdint.h>
#include <openssl/sha.h>

#include "uint256.h"
#include "sph/sph_groestl.h"

#include "miner.h"
#include <cuda_runtime.h>

static bool init[MAX_GPUS] = { 0 };
static uint32_t *h_found[MAX_GPUS];

void myriadgroestl_cpu_init(int thr_id, uint32_t threads);
void myriadgroestl_cpu_setBlock(int thr_id, void *data, void *pTargetIn);
void myriadgroestl_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *nounce);

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

extern "C" void myriadhash(void *state, const void *input)
{
	uint32_t hashA[16], hashB[16];
	sph_groestl512_context ctx_groestl;
	SHA256_CTX sha256;

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, input, 80);
	sph_groestl512_close(&ctx_groestl, hashA);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256,(unsigned char *)hashA, 64);
	SHA256_Final((unsigned char *)hashB, &sha256);

	memcpy(state, hashB, 32);
}

extern "C" int scanhash_myriad(int thr_id, uint32_t *pdata, uint32_t *ptarget,
	uint32_t max_nonce, uint32_t *hashes_done)
{
	uint32_t start_nonce = pdata[19]++;
	uint32_t throughput = device_intensity(thr_id, __func__, 1 << 17);
	throughput = min(throughput, max_nonce - start_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x0000ff;

	// init
	if(!init[thr_id])
	{
#if BIG_DEBUG
#else
		myriadgroestl_cpu_init(thr_id, throughput);
#endif
		cudaMallocHost(&(h_found[thr_id]), 4 * sizeof(uint32_t));
		init[thr_id] = true;
	}

	uint32_t endiandata[32];
	for (int kk=0; kk < 32; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	// Context mit dem Endian gedrehten Blockheader vorbereiten (Nonce wird sp�ter ersetzt)
	myriadgroestl_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

	do {
		const uint32_t Htarg = ptarget[7];

		myriadgroestl_cpu_hash(thr_id, throughput, pdata[19], h_found[thr_id]);

		if (h_found[thr_id][0] != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], h_found[thr_id][0]);
			myriadhash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - start_nonce + throughput;
				if (h_found[thr_id][1] != 0xffffffff)
				{
					be32enc(&endiandata[19], h_found[thr_id][1]);
					myriadhash(vhash64, endiandata);
					if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{

						pdata[21] = h_found[thr_id][1];
						res++;
						if (opt_benchmark)
							applog(LOG_INFO, "GPU #%d Found second nounce %08x", thr_id, h_found[thr_id][1], vhash64[7], Htarg);
					}
					else
					{
						if (vhash64[7] != Htarg)
						{
							applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", thr_id, h_found[thr_id][1]);
						}
					}

				}
				pdata[19] = h_found[thr_id][0];
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, h_found[thr_id][0], vhash64[7], Htarg);
				return res;
			}
			else
			{
				if (vhash64[7] != Htarg)
				{
					applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", thr_id, h_found[thr_id][0]);
				}
			}
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - start_nonce + 1;
	return 0;
}

