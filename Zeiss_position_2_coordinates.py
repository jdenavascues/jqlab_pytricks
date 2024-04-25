import os
import pandas as pd
pd.set_option('mode.sim_interactive', True)
import numpy as np
from itertools import chain

folderpath = os.path.join(os.getcwd(),'LSM data')

def get_positions(folderpath):
    '''Identifies positions file, reads it, converts positions and sorts them per gut'''

    # identify position file
    posfilelist = np.sort([x for x in os.listdir(folderpath) if x.endswith('.pos')]).tolist()
    
    positions = pd.DataFrame()
    
    for posfile in posfilelist:
        # read positions from files
        f = open(os.path.join(folderpath,posfile),'rU')
        print 'reading file: ', posfile
        lines = f.readlines()
        
        pos_record = []
        try:
            del x, y, z
        except:
            pass
        for line in lines:
            if line.startswith('\t\tX = '):
                x = float(line.split('= ')[1].split(' \xb5m')[0])
            if line.startswith('\t\tY = '):
                y = float(line.split('= ')[1].split(' \xb5m')[0])
            if line.startswith('\t\tZ = '):
                z = float(line.split('= ')[1].split(' \xb5m')[0])
            try:
                pos_record.append([x, y, z])
                print 'recording positions: ', x, y, z
                del x, y, z
            except:
                'something went wrong'
        
        pos_record = np.array(pos_record)
        
        tuples = (posfile.split('_')[0], posfile.split('.')[0][-1])
        tuples = [tuples, tuples, tuples]
        tuples = list(zip(np.array(tuples)[:,0].tolist(), np.array(tuples)[:,1].tolist(), ['x','y','z']))
        posindex = pd.MultiIndex.from_tuples(tuples, names=['date','attempt','coordinate'])

        positions = pd.concat([positions, pd.DataFrame(pos_record,columns=posindex)], axis=1)
        
    return positions

positions = get_positions(folderpath)
session1 = positions.iloc[ :, positions.columns.get_level_values(0) == positions.columns.levels[0][0] ]
session2 = positions.iloc[ :, positions.columns.get_level_values(0) == positions.columns.levels[0][1] ]

a1 = np.array(session1['20190115','1','z'])
a2 = np.array(session1['20190115','2','z'])

b1 = np.array(session2['20190121','1','z'])
b2 = np.array(session2['20190121','2','z'])

writer = pd.ExcelWriter('coordinates.xlsx')
positions.to_excel(writer, header=True)

