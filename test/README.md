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

More optimizations

```
th -lrnn -e 'rnn.bigtest({"LSTM","GRU"})'
Running 3 tests
1/3 LSTM_char_rnn ....................................................... [PASS]
2/3 GRU ................................................................. [WAIT]CPU test
old GRU time: 0.039725697040558 seconds
step GRU time: 0.014464259147644 seconds
luarec GRU time: 0.017707204818726 seconds
rec GRU time: 0.013900947570801 seconds
luaseq GRU time: 0.016570293903351 seconds
seq GRU time: 0.012663447856903 seconds
RecGRU-C 1.2738127907136 faster than RecGRU-Lua
RecGRU 2.8577690001509 faster than old GRU
SeqGRU 1.0977221786579 faster than RecGRU
SeqGRU-C 1.3085136126113 faster than SeqGRU-Lua
Memory test
old GRU memory: 82.804834365845:1.833381652832 MB
step GRU memory: 10.018351554871:1.5651426315308 MB
rec GRU memory: 10.018255233765:1.5337238311768 MB
seq GRU memory: 6.3827362060547:1.5385322570801 MB
2/3 GRU ................................................................. [PASS]
3/3 LSTM ................................................................ [WAIT]CPU test
fast LSTM time: 0.044381546974182 seconds
step LSTM time: 0.021313452720642 seconds
luarec LSTM time: 0.021889448165894 seconds
rec LSTM time: 0.017923295497894 seconds
luaseq LSTM time: 0.018705642223358 seconds
seq LSTM time: 0.016467046737671 seconds
RecLSTM-C 1.2212847893104 faster than RecLSTM-Lua
RecLSTM 2.476193453341 faster than FastLSTM
SeqLSTM 1.0884341183591 faster than RecLSTM
SeqLSTM-C 1.1359439565181 faster than SeqLSTM-Lua
Memory test
fast LSTM memory: 98.2790184021:2.2749843597412 MB
step LSTM memory: 17.168484687805:2.1293544769287 MB
rec LSTM memory: 13.375099182129:2.0412521362305 MB
seq LSTM memory: 8.8264684677124:2.0093183517456 MB
3/3 LSTM ................................................................ [PASS]
Completed 0 asserts in 3 tests with 0 failures and 0 errors and 1 warning
--------------------------------------------------------------------------------
```