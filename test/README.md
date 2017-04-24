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

Moving StepLSTM to C and changing inputsize, outputsize = 512, 512 to 128, 128 (emphasizes non-BLAS overhead).

```
fast LSTM time: 0.12676219940186 seconds
step LSTM time: 0.083592581748962 seconds
luarec LSTM time: 0.078796887397766 seconds
rec LSTM time: 0.068624520301819 seconds
seq LSTM time: 0.06977710723877 seconds
RecLSTM-C 1.1482322506767 faster than RecLSTM-Lua
RecLSTM 1.8471852166593 faster than FastLSTM
SeqLSTM 0.98348187560991 faster than RecLSTM
Memory test
fast LSTM memory: 98.27904510498:2.2750110626221 MB
step LSTM memory: 17.33301448822:2.1554565429688 MB
rec LSTM memory: 13.508204460144:2.0672521591187 MB
seq LSTM memory: 8.9209499359131:3.0412549972534 MB
```