---
title: ANN Report
author: Carson Fox
date: September, 2021
---
## Problem 3

### 284xnx10
| eta   | n = 10   | 25    | 50    |
| ----- | -------- | ----- | ----- |
| .5    | .90      | .92   | .93   |
| .25   | .89      | .91   | .74   |
| .125  | .86      | .80   | .72   |

### 284x10xnx10
| eta   | n = 10   | 25    | 50    |
| ----- | -------- | ----- | ----- |
| .5    | .89      | .91   | .91   |
| .25   | .88      | .89   | .88   |
| .125  | .81      | .86   | .87   |

### 284x25xnx10
| eta   | n = 10   | 25    | 50    |
| ----- | -------- | ----- | ----- |
| .5    | .91      | .92   | .93   |
| .25   | .90      | .90   | .92   |
| .125  | .87      | .89   | .90   |

### 284x50xnx10
| eta   | n = 10   | 25    | 50    |
| ----- | -------- | ----- | ----- |
| .5    | .92      | .93   | .93   |
| .25   | .91      | .91   | .92   |
| .125  | .88      | .89   | .90   |

### Observations
It seems a higher learning rate, unsuprisingly, causes the networks to learn faster.
Much larger networks were not significantly more accurate when compared to anything bigger than about 10x10, so it seems like 25x25 is a good choice.
This makes sense, since that number of parameters is close to the number of parameters in the input.
