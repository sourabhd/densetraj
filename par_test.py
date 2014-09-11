
# coding: utf-8

# In[87]:

from IPython import parallel

c = parallel.Client(profile='sge', sshserver='sdaptardar@130.245.4.230')
view = c[:]
c.ids


# In[88]:

get_ipython().magic(u"px print('Hello World!')")


# In[89]:

get_ipython().run_cell_magic(u'px', u'', u'import os\nimport socket\nprint os.getpid()\nprint socket.gethostname()')


# In[90]:

A = 'Shared var '
get_ipython().magic(u"px A = 'My var'")


# In[91]:

def myfunc(x):
    import os
    import socket
    return 'I am :', A, os.getpid(), ' on ', socket.gethostname(), ' running: ', str(x)
    
view.block = False

part = view.scatter('part', range(24))
print(view['part'])

res = view.map(myfunc, view['part'])


# In[92]:

import time
while not res.ready():
    time.sleep(1)
    print res.progress


# In[93]:

print 'Ready: ', res.ready()
print 'Progress: ', res.progress


# In[94]:

res.result


# In[94]:



