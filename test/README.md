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
fast LSTM time: 0.12144248485565 seconds
step LSTM time: 0.073302102088928 seconds
luarec LSTM time: 0.075324392318726 seconds
rec LSTM time: 0.066254210472107 seconds
luaseq LSTM time: 0.067860889434814 seconds
seq LSTM time: 0.062430810928345 seconds
RecLSTM-C 1.1368997046676 faster than RecLSTM-Lua
RecLSTM 1.8329776174267 faster than FastLSTM
SeqLSTM 1.0612421893438 faster than RecLSTM
SeqLSTM-C 1.0869775424302 faster than SeqLSTM-Lua
Memory test
fast LSTM memory: 98.27904510498:2.2750110626221 MB
step LSTM memory: 17.168065071106:2.1289348602295 MB
rec LSTM memory: 13.374607086182:2.0407600402832 MB
seq LSTM memory: 8.8895826339722:3.0098876953125 MB
```