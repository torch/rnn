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

Moving StepLSTM and SeqLSTM to C and changing inputsize, outputsize = 512, 512 to 128, 128 (emphasizes non-BLAS overhead).

```
fast LSTM time: 0.066211605072021 seconds
step LSTM time: 0.036829793453217 seconds
luarec LSTM time: 0.038231909275055 seconds
rec LSTM time: 0.033363950252533 seconds
luaseq LSTM time: 0.035267758369446 seconds
seq LSTM time: 0.031274902820587 seconds
RecLSTM-C 1.1459047560519 faster than RecLSTM-Lua
RecLSTM 1.9845253505914 faster than FastLSTM
SeqLSTM 1.0667962885106 faster than RecLSTM
SeqLSTM-C 1.1276696388719 faster than SeqLSTM-Lua
Memory test
fast LSTM memory: 98.27904510498:2.2750110626221 MB
step LSTM memory: 17.168065071106:2.1289348602295 MB
rec LSTM memory: 13.374607086182:2.0407600402832 MB
seq LSTM memory: 8.8895826339722:3.0098876953125 MB
```