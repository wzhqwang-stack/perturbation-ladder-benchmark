# T3 网格汇总（stage02，自动生成）

生成时间：2026-11-02 23:28

## 全格矩阵（R²）

|                                                 |    149 |    150 |    151 |
|:------------------------------------------------|-------:|-------:|-------:|
| ('adamson', 'gears')                            | 0.6915 | 0.7606 | 0.7025 |
| ('adamson', 'linear')                           | 0.6697 | 0.712  | 0.6264 |
| ('adamson', 'mean_passthrough')                 | 0.3251 | 0.3428 | 0.2892 |
| ('adamson', 'random')                           | 0.715  | 0.6491 | 0.4718 |
| ('adamson', 'scgen')                            | 0.3259 | 0.4925 | 0.3227 |
| ('norman', 'gears')                             | 0.5932 | 0.6379 | 0.7691 |
| ('norman', 'linear')                            | 0.3091 | 0.4145 | 0.4938 |
| ('norman', 'mean_passthrough')                  | 0.1672 | 0.2611 | 0.3436 |
| ('norman', 'random')                            | 0.216  | 0.3432 | 0.3175 |
| ('norman', 'scgen')                             | 0.1906 | 0.2796 | 0.295  |
| ('replogle_k562_essential', 'gears')            | 0.659  | 0.677  | 0.7128 |
| ('replogle_k562_essential', 'linear')           | 0.5768 | 0.5758 | 0.634  |
| ('replogle_k562_essential', 'mean_passthrough') | 0.4733 | 0.4686 | 0.5385 |
| ('replogle_k562_essential', 'random')           | 0.4891 | 0.496  | 0.5468 |
| ('replogle_k562_essential', 'scgen')            | 0.3622 | 0.3038 | 0.3261 |

## 全格矩阵（Pearson）

|                                                 |    149 |    150 |    151 |
|:------------------------------------------------|-------:|-------:|-------:|
| ('adamson', 'gears')                            | 0.906  | 0.9133 | 0.8976 |
| ('adamson', 'linear')                           | 0.9071 | 0.9112 | 0.8908 |
| ('adamson', 'mean_passthrough')                 | 0.8124 | 0.8239 | 0.8013 |
| ('adamson', 'random')                           | 0.9162 | 0.8974 | 0.8553 |
| ('adamson', 'scgen')                            | 0.8234 | 0.8443 | 0.8174 |
| ('norman', 'gears')                             | 0.8535 | 0.8706 | 0.903  |
| ('norman', 'linear')                            | 0.791  | 0.8155 | 0.8231 |
| ('norman', 'mean_passthrough')                  | 0.7435 | 0.7576 | 0.7643 |
| ('norman', 'random')                            | 0.7625 | 0.8022 | 0.7737 |
| ('norman', 'scgen')                             | 0.7712 | 0.7688 | 0.7772 |
| ('replogle_k562_essential', 'gears')            | 0.8907 | 0.8871 | 0.9019 |
| ('replogle_k562_essential', 'linear')           | 0.8674 | 0.8617 | 0.8825 |
| ('replogle_k562_essential', 'mean_passthrough') | 0.8419 | 0.8365 | 0.8592 |
| ('replogle_k562_essential', 'random')           | 0.8419 | 0.839  | 0.8587 |
| ('replogle_k562_essential', 'scgen')            | 0.8118 | 0.8133 | 0.8305 |

## GEARS vs linear 配对 Wilcoxon（逐条件差值；p_BH=全表 BH 校正）

| dataset                 |   seed | metric      |   n |   median_diff |   win_rate |      p |   p_BH | sig_BH<0.05   |
|:------------------------|-------:|:------------|----:|--------------:|-----------:|-------:|-------:|:--------------|
| norman                  |    149 | r2_deg      | 116 |        0.14   |     0.8017 | 0      | 0      | yes           |
| norman                  |    149 | pearson_deg | 116 |        0.0386 |     0.7155 | 0      | 0      | yes           |
| norman                  |    150 | r2_deg      | 116 |        0.105  |     0.6897 | 0      | 0      | yes           |
| norman                  |    150 | pearson_deg | 116 |        0.0294 |     0.7069 | 0      | 0      | yes           |
| norman                  |    151 | r2_deg      | 121 |        0.1142 |     0.8099 | 0      | 0      | yes           |
| norman                  |    151 | pearson_deg | 121 |        0.0375 |     0.7851 | 0      | 0      | yes           |
| adamson                 |    149 | r2_deg      |  21 |       -0.0046 |     0.3333 | 0.2877 | 0.3699 | no            |
| adamson                 |    149 | pearson_deg |  21 |       -0.0019 |     0.3333 | 0.229  | 0.3171 | no            |
| adamson                 |    150 | r2_deg      |  21 |       -0.0194 |     0.381  | 0.8917 | 0.8917 | no            |
| adamson                 |    150 | pearson_deg |  21 |       -0.0096 |     0.381  | 0.412  | 0.4944 | no            |
| adamson                 |    151 | r2_deg      |  21 |       -0.0155 |     0.381  | 0.8117 | 0.8917 | no            |
| adamson                 |    151 | pearson_deg |  21 |       -0.0053 |     0.381  | 0.8649 | 0.8917 | no            |
| replogle_k562_essential |    149 | r2_deg      | 272 |        0.0209 |     0.6324 | 0      | 0      | yes           |
| replogle_k562_essential |    149 | pearson_deg | 272 |        0.0053 |     0.6066 | 0      | 0      | yes           |
| replogle_k562_essential |    150 | r2_deg      | 272 |        0.0258 |     0.6397 | 0      | 0      | yes           |
| replogle_k562_essential |    150 | pearson_deg | 272 |        0.0069 |     0.625  | 0      | 0      | yes           |
| replogle_k562_essential |    151 | r2_deg      | 272 |        0.0137 |     0.6581 | 0      | 0      | yes           |
| replogle_k562_essential |    151 | pearson_deg | 272 |        0.0045 |     0.6213 | 0      | 0      | yes           |
