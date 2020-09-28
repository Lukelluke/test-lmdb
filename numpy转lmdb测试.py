import numpy as np
import lmdb

def write_lmdb(filename):
    print ('Write lmdb')

    lmdb_env = lmdb.open(filename, map_size=int(1e9))

    n_samples= 2
    X= (255*np.random.rand(n_samples,2)).astype(np.float32)
    print("type(X) = ", type(X))                    # type(X) =  <class 'numpy.ndarray'>
    print("type(X[0][0]) = ", type(X[0][0]))        # type(X[0][0]) =  <class 'numpy.float32'>
    print("line 11 "+str(X))
    y= np.random.rand(n_samples).astype(np.float32)
    print("line 13 " + str(y))

    for i in range(n_samples):
        with lmdb_env.begin(write=True) as lmdb_txn:
            lmdb_txn.put(('X_'+str(i)).encode(), X)
            lmdb_txn.put(('y_'+str(i)).encode(), y)

            print( 'X:',X)
            print ('y:',y)

def read_lmdb(filename):
    print ('Read lmdb')

    lmdb_env = lmdb.open(filename)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    n_samples=0
    with lmdb_env.begin() as lmdb_txn:
        with lmdb_txn.cursor() as lmdb_cursor:
            for key, value in lmdb_cursor:
                print(key)
                if(('X').encode() in key):
                    print(np.fromstring(value, dtype=np.float32))
                    tmp = np.fromstring(value, dtype=np.float32)
                    print("tmp = ", tmp)
                    tmp = np.reshape(tmp, (2, 2))
                    print("new tmp = ", tmp)

                if(('y').encode() in key):
                    print(np.fromstring(value, dtype=np.float32))

                n_samples=n_samples+1

    print('n_samples',n_samples)

write_lmdb('temp.db')
read_lmdb('temp.db')

'''
type(X) =  <class 'numpy.ndarray'>
type(X[0][0]) =  <class 'numpy.float32'>
line 11 [[ 59.600163 225.7832  ]
 [136.14442   73.89465 ]]
line 13 [0.34790182 0.5339572 ]
X: [[ 59.600163 225.7832  ]
 [136.14442   73.89465 ]]
y: [0.34790182 0.5339572 ]
X: [[ 59.600163 225.7832  ]
 [136.14442   73.89465 ]]
y: [0.34790182 0.5339572 ]
Read lmdb
b'X_0'
[ 59.600163 225.7832   136.14442   73.89465 ]
tmp =  [ 59.600163 225.7832   136.14442   73.89465 ]
new tmp =  [[ 59.600163 225.7832  ]
 [136.14442   73.89465 ]]
b'X_1'
[ 59.600163 225.7832   136.14442   73.89465 ]
tmp =  [ 59.600163 225.7832   136.14442   73.89465 ]


[[ 59.600163 225.7832  ]
 [136.14442   73.89465 ]]
b'y_0'
[0.34790182 0.5339572 ]
b'y_1'
[0.34790182 0.5339572 ]
n_samples 4
'''