./YCSB/bin/ycsb.sh load basic -P configs/configA.dat -threads 100 > YCSB_data/YCSB_A_prefill
./YCSB/bin/ycsb.sh load basic -P configs/configB.dat -threads 100 > YCSB_data/YCSB_B_prefill
./YCSB/bin/ycsb.sh load basic -P configs/configC.dat -threads 100 > YCSB_data/YCSB_C_prefill
./YCSB/bin/ycsb.sh load basic -P configs/configD.dat -threads 100  > YCSB_data/YCSB_D_prefill
./YCSB/bin/ycsb.sh load basic -P configs/configE.dat -threads 100  > YCSB_data/YCSB_E_prefill
