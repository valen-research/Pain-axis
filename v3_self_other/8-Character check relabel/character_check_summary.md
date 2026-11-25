# Character check, one rubric for both runs

Labels: in = in character, out = out of character, unclear = can't tell, garbled. Percent of replies (20 per model and character). 'kept' = in / (in + out), among replies that show an identity. Base = 11 models, instruct = 14 (Qwen 3 8B and 14B counted as instruct).


## Run A (new 420)

Total replies: 4500; in 1306, out 346, unclear 1601, garbled 1247


### base models (n=11)

```
label             in  out  unclear  garbled  kept
character                                        
self            29.0  1.0     41.0     30.0  97.0
claude          23.0  2.0     49.0     25.0  91.0
aria            22.0  3.0     41.0     34.0  88.0
maria            6.0  0.0     31.0     63.0  93.0
curie           15.0  2.0     34.0     50.0  89.0
rex             11.0  2.0     33.0     54.0  83.0
teapot           9.0  2.0     28.0     61.0  83.0
jack_masochist  13.0  4.0     41.0     42.0  78.0
jack_sadist     13.0  3.0     37.0     47.0  83.0
```


### instruct models (n=14)

```
label             in   out  unclear  garbled  kept
character                                         
self            51.0   5.0     41.0      4.0  92.0
claude          51.0   5.0     42.0      2.0  91.0
aria            47.0  11.0     36.0      6.0  81.0
maria           28.0  14.0     42.0     16.0  66.0
curie           54.0  11.0     20.0     16.0  83.0
rex             39.0  11.0     34.0     16.0  78.0
teapot          38.0  16.0     25.0     21.0  70.0
jack_masochist  22.0  17.0     38.0     23.0  57.0
jack_sadist     26.0  19.0     33.0     22.0  58.0
```


## Run B (original 420 with AI-specific harms)

Total replies: 4500; in 1284, out 357, unclear 2014, garbled 845


### base models (n=11)

```
label             in  out  unclear  garbled  kept
character                                        
self            27.0  0.0     58.0     15.0  98.0
claude          22.0  1.0     65.0     11.0  96.0
aria            22.0  1.0     60.0     17.0  94.0
maria           11.0  5.0     34.0     50.0  67.0
curie           13.0  3.0     45.0     40.0  83.0
rex             19.0  3.0     33.0     45.0  87.0
teapot          19.0  1.0     30.0     50.0  95.0
jack_masochist  17.0  5.0     36.0     41.0  76.0
jack_sadist     19.0  8.0     32.0     40.0  70.0
```


### instruct models (n=14)

```
label             in   out  unclear  garbled  kept
character                                         
self            39.0   2.0     58.0      0.0  94.0
claude          37.0   4.0     59.0      0.0  90.0
aria            39.0   7.0     52.0      3.0  85.0
maria           28.0  14.0     52.0      6.0  67.0
curie           49.0  15.0     31.0      6.0  77.0
rex             38.0  12.0     43.0      7.0  76.0
teapot          41.0  14.0     34.0     10.0  74.0
jack_masochist  27.0  18.0     40.0     15.0  60.0
jack_sadist     29.0  19.0     40.0     12.0  60.0
```
