# Tests and benchmarks

## LSTM benchmarks

```
th -lrnn -e 'rnn.bigtest("LSTM")'
CPU test
fast LSTM time: 0.24093070030212 seconds
step LSTM time: 0.18469240665436 seconds
rec LSTM time: 0.1736387014389 seconds
seq LSTM time: 0.17547559738159 seconds
RecLSTM 1.387540325432 faster than FastLSTM
SeqLSTM 0.9895319009019 faster than RecLSTM
Memory test
fast LSTM memory: 415.27123260498:32.298448562622 MB
step LSTM memory: 92.378144264221:32.339014053345 MB
rec LSTM memory: 77.209584236145:32.063309669495 MB
seq LSTM memory: 59.674856185913:36.158442497253 MB
```