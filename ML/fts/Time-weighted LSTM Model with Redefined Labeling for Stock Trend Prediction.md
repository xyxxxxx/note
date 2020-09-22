**keyword**

LSTM, time-weighted

**key contents**

趋势的定义：

+ 上涨趋势：序列$$\{P_i\},\{T_i\}$$是递增的；序列$$\{P_i\},\{T_i\}$$的规模均大于$$K$$；$$\forall i$$，满足
  $$
  P_i-T_i< \alpha(P_i-T_{i-1})
  $$

+ 

+ 下跌趋势：序列$$\{P_i\},\{T_i\}$$是递减的；序列$$\{P_i\},\{T_i\}$$的规模均大于$$K$$；$$\forall i$$，满足
  $$
  P_i-T_i>(1+\beta)(P_i-T_{i-1})
  $$

+ 无趋势：不属于以上两种趋势的趋势

其中$$P_i$$表示极大值(peak)，$$T_i$$表示极小值(trough)，$$T_i$$发生于$$P_i$$和$$P_{i+1}$$之间。

