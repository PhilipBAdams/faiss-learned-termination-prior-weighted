/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */



#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iterator>
#include <vector>
#include <chrono>
#include <omp.h>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/time.h>

#include "../ProductQuantizer.h"
#include "../IndexHNSW.h"
#include "../IndexPQ.h"

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/


float * fvecs_read (const char *fname,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}



int main(int argc, char* argv[])
{
    double t0 = elapsed();

    // this is typically the fastest one.
    //const char *index_key = "IVF4096,Flat";

    // these ones have better memory usage
    // const char *index_key = "Flat";
    // const char *index_key = "PQ32";
    // const char *index_key = "PCA80,Flat";
    // const char *index_key = "IVF4096,PQ8+16";
    // const char *index_key = "IVF4096,PQ32";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

    faiss::IndexHNSWMultiPQ* index;

	if (argc != 17)
	{
		 printf("Usage: %s data_file train_low_file train_high_file query_file prior_file gt_file threshold hnsw_m pq_m nbits_low nbits_high k efSearch strategy niter outfile\n", argv[0]);
		 exit(1);
	}
	
	char* data_file = argv[1];
	char* train_low_file = argv[2];
	char* train_high_file = argv[3];
	char* query_file = argv[4];
	char* prior_file = argv[5];
	char* gt_file = argv[6];
	float threshold = atof(argv[7]);
	size_t hnsw_m = atoi(argv[8]);
	size_t pq_m = atoi(argv[9]);
	size_t nbits_low = atoi(argv[10]);
	size_t nbits_high = atoi(argv[11]);
	size_t k = atoi(argv[12]);
	size_t efSearch = atoi(argv[13]);
	char* strategy = argv[14];
	size_t niter = atoi(argv[15]);
	char* outfile = argv[16];
	printf("threshold: %f, hnsw_m: %d, pq_m: %d, nbits_low: %d, nbits_high: %d, k: %d, efSearch: %d, niter: %d\n", threshold, hnsw_m, pq_m, nbits_low, nbits_high, k, efSearch, niter);

	size_t d;

	{
		 printf ("[%.3f s] Loading train_low set\n", elapsed() - t0);

		 size_t nt;
		 float *xt = fvecs_read(train_low_file, &d, &nt);

		 printf ("[%.3f s] Preparing index d=%ld\n",
				 elapsed() - t0, d);
		 index = new faiss::IndexHNSWMultiPQ(d, pq_m, nbits_low, nbits_high, threshold, hnsw_m);
		 index->hnsw.efSearch = efSearch;
		 printf ("[%.3f s] Training low on %ld vectors\n", elapsed() - t0, nt);

		 index->train(nt, xt);
		 delete [] xt;
	}

	{
		 printf ("[%.3f s] Loading train_high set\n", elapsed() - t0);

		 size_t nth, dh;
		 float *xt = fvecs_read(train_high_file, &dh, &nth);
		 assert(d == dh || !"train sets have different dimension");

		 printf ("[%.3f s] Training high on %ld vectors\n", elapsed() - t0, nth);

		 ((faiss::IndexMultiPQ*) index->storage)->train_high_precision(nth, xt);
		 ((faiss::IndexMultiPQ*) index->storage)->mpq.compute_sdc_table();
		 delete [] xt;
	}

	{
		 printf ("[%.3f s] Loading prior set\n", elapsed() - t0);

		 size_t np, dp;
		 float *xp = fvecs_read(prior_file, &dp, &np);
		 

		 printf ("[%.3f s] Adding priors\n", elapsed() - t0);
		 index->set_priors(np, xp, strategy);
		 ((faiss::IndexMultiPQ*) index->storage)->add_priors(np, std::vector<float>(xp, xp + np));
		 //delete [] xp;
	}


	{
		 printf ("[%.3f s] Loading database\n", elapsed() - t0);

		 size_t nb, d2;
		 float *xb = fvecs_read(data_file, &d2, &nb);
		 assert(d == d2 || !"dataset does not have same dimension as train set");

		 printf ("[%.3f s] Indexing database, size %ld*%ld\n",
				 elapsed() - t0, nb, d);

		 index->add(nb, xb);

		 delete [] xb;
	}

	size_t nq;
	float *xq;

	{
		 printf ("[%.3f s] Loading queries\n", elapsed() - t0);

		 size_t d2;
		 xq = fvecs_read(query_file, &d2, &nq);
		 assert(d == d2 || !"query does not have same dimension as train set");

	}

	
	faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors
	size_t k2;
	{
		 printf ("[%.3f s] Loading ground truth for %ld queries\n",
				 elapsed() - t0, nq);

		 // load ground-truth and convert int to long
		 size_t nq2;

		 int *gt_int = ivecs_read(gt_file, &k2, &nq2);
		 assert(nq2 == nq || !"incorrect nb of ground truth entries");

		 gt = new faiss::Index::idx_t[k2 * nq];
		 for(int i = 0; i < k2 * nq; i++) {
			  gt[i] = gt_int[i];
		 }
		 delete [] gt_int;
	}

	std::ofstream ofile;
	ofile.open(outfile);
	for (efSearch = 10; efSearch <=100; efSearch += 10)
	{ // Use the found configuration to perform a search
		 index->hnsw.efSearch = efSearch;
		 printf ("[%.3f s] Perform a search on %ld queries\n",
				 elapsed() - t0, nq);

		 // output buffers
		 faiss::Index::idx_t *I = new  faiss::Index::idx_t[nq * k];
		 float *D = new float[nq * k];
		 double start = omp_get_wtime();
		 for (int i = 0; i < niter; i++) {
		 	index->search(nq, xq, k, D, I);
		 }
		 double end = omp_get_wtime();
		 printf("Search Time: %f s\n", end-start);

		 printf ("[%.3f s] Compute recalls\n", elapsed() - t0);

// evaluate result by hand.
		 float recalls[k];
		 for (int i = 0; i < k; i++) {
			 recalls[i] = 0;
		 }
		 for(int i = 0; i < nq; i++) {
			  int gt_nn = gt[i * k2];
			  for(int j = 0; j < k; j++) {
				   if (I[i * k + j] == gt_nn) {
						for (int i2 = j; i2 < k; i2++) {
							recalls[i2]++;
						}
				   }
			  }
		 }
		 ofile << efSearch << "," << (end-start) / ((double) niter);
		 for (int i = 0; i < k; i++) {
			 ofile << "," << (recalls[i] / float(nq));
		 }
		 ofile << std::endl;

		 printf("R@1 = %.4f\n", recalls[0] / float(nq));
		 printf("R@10 = %.4f\n", recalls[9] / float(nq));
		 printf("R@100 = %.4f\n", recalls[99] / float(nq));
	}
	ofile.close();

	

	delete [] xq;
    delete [] gt;
    delete index;
    return 0;
}
