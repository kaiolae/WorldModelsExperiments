for i in `seq 1 10`;
do
python rnn_train.py --epochs 1000 --num_mixtures 1 --output_file_name run$i
python rnn_train.py --epochs 1000 --num_mixtures 2 --output_file_name run$i
python rnn_train.py --epochs 1000 --num_mixtures 4 --output_file_name run$i
python rnn_train.py --epochs 1000 --num_mixtures 8 --output_file_name run$i
python rnn_train.py --epochs 1000 --num_mixtures 16 --output_file_name run$i
done
