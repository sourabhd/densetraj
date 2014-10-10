from __future__ import print_function
import pandas as pd
from ggplot import *
df = pd.read_csv('/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj/train/actioncliptrain00351.txt', delim_whitespace=True, header=None)
max_t =  df.loc[:,0].max()
print('max: ' + str(max_t))
p = ggplot(aes(x=0), data=pd.DataFrame(df.iloc[:,0])) + geom_histogram(binwidth=1) + xlim(0,max_t)
#print(p)
ggsave(plot=p, filename='hist.png', height=5, units='cm')
