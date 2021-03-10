#! /bin/bash

for hnsw_m in 16 32
do
    for nbits_low in 4 8
    do
        for nbits_high in 8 12
        do 
            ./hnsw_multipq_test /data/ann_proj/gist/gist_base.fvecs /data/ann_proj/gist/gist_learn.fvecs /data/ann_proj/gist/gist_learn.fvecs /data/ann_proj/gist/gist_query.fvecs /data/ann_proj/gist/gist-priors-expfalloff.fvecs /data/ann_proj/gist/gist_groundtruth.ivecs 0.0 $hnsw_m 32 $nbits_low $nbits_high 100 100 Random 100 data/HNSW${hnsw_m}_MULTIPQ_32x${nbits_low}_${nbits_high}_Random.csv
            ./hnsw_multipq_test /data/ann_proj/gist/gist_base.fvecs /data/ann_proj/gist/gist_learn.fvecs /data/ann_proj/gist/gist_learn.fvecs /data/ann_proj/gist/gist_query.fvecs /data/ann_proj/gist/gist-priors-expfalloff.fvecs /data/ann_proj/gist/gist_groundtruth.ivecs 0.0 $hnsw_m 32 $nbits_low $nbits_high 100 100 PriorSum 100 data/HNSW${hnsw_m}_MULTIPQ_32x${nbits_low}_${nbits_high}_PriorSum.csv
            ./hnsw_pq_test /data/ann_proj/gist/gist_base.fvecs /data/ann_proj/gist/gist_learn.fvecs /data/ann_proj/gist/gist_learn.fvecs /data/ann_proj/gist/gist_query.fvecs /data/ann_proj/gist/gist-priors-expfalloff.fvecs /data/ann_proj/gist/gist_groundtruth.ivecs 0.0 $hnsw_m 32 $nbits_low $nbits_high 100 100 Random 100 data/HNSW${hnsw_m}_PQ_32x${nbits_low}_Random.csv
            ./hnsw_pq_test /data/ann_proj/gist/gist_base.fvecs /data/ann_proj/gist/gist_learn.fvecs /data/ann_proj/gist/gist_learn.fvecs /data/ann_proj/gist/gist_query.fvecs /data/ann_proj/gist/gist-priors-expfalloff.fvecs /data/ann_proj/gist/gist_groundtruth.ivecs 0.0 $hnsw_m 32 $nbits_low $nbits_high 100 100 PriorSum 100 data/HNSW${hnsw_m}_PQ_32x${nbits_low}_PriorSum.csv
        done
    done
done