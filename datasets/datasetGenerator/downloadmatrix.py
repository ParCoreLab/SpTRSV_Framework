import subprocess
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

class DownloadMatrix:

    def __init__(self):
        pass

    def downloadSSGET(self, ID):
        #print(ID)
        result = subprocess.run(['ssget', '-e', '-i', ID], stdout=subprocess.PIPE)
        path = result.stdout.decode().strip()
        return [result.returncode, path]

    def ReadMM(self,path, mat_type):
        GBs = 512
        chunksize = GBs * 1024 * 1024 
        lines = 0
        try:
            mmfile = open(path, 'r')

        except IOError:
            print("File not found")
            return "File not found"
        header = mmfile.readline()
        lines = lines + 1

        if len(header) == 0:
            print("Empty file.")
            return
        header = header.split()
        head0 = header[0].lower()
        head1 = header[1].lower()
        rep = header[2].lower()
        field = header[3].lower()
        symm = header[4].lower()

        if len(symm) == 0:
            print('Not enough words in header line of file {}'.format(path))
            print('Recognized format: ')
            print('%%MatrixMarket matrix representation field symmetry')
            print('Check header line.')

        commentline = mmfile.readline()
        lines = lines + 1
        while len(commentline) > 0 and commentline[0] == '%':
            commentline = mmfile.readline()
            lines = lines + 1

        commentline = commentline.split()
        rows = commentline[0]
        cols = commentline[1]
        entries = commentline[2]
        print('Entries: {}'.format(entries))
        if rep == 'coordinate':
            while entries == 0:
                commentline = mmfile.readline()
                lines = lines + 1
                if len(commentline) == 0:
                    print('End-of-file reached before size information was found.')
                commentline = commentline.split()
                if len(commentline) > 0 and len(commentline) != 3:
                    print('Invalid size specification line')
                    return
            rows = int(commentline[0])
            cols = int(commentline[1])
            entries = int(commentline[2])
            matdata_coo = np.array([])
            cnt = 0
            nlc = 0      # new line characters
            extra_line = 0
            if field == 'real':
                #T = mmfile.readlines(chunksize)
                #while T != None:
                T = mmfile.read(chunksize)
                readlength = len(T)
                last_char = T[:-1]
                while T != '':
                    T = T.split()
                    mat_str = np.array(T)
                    T = None
                    cnt = cnt + 1
                    print ('{0:0.1f} GB processed \r'.format(readlength*cnt/1024), end=' ')                   
                    if readlength == chunksize:
                        #seekback = len(T[-1].encode('utf-8'))
                        #mmfile.seek(seekback)
                        extra_line = len(mat_str) % 3
                        seekback = 0
                        if extra_line == 0:
                            seekback = len(mat_str[-1].encode('utf-8')) + len(mat_str[-2].encode('utf-8')) \
                                       + len(mat_str[-3].encode('utf-8'))
                            if last_char == '\n':
                                nlc = 3
                            else:
                                nlc = 2
                            matdata_coo = np.append(matdata_coo, mat_str[:-3].astype(np.float))
                        if extra_line == 1:
                            seekback = len(mat_str[-1].encode('utf-8'))
                            if last_char == '\n':
                                nlc = 1
                            else:
                                nlc = 0
                            matdata_coo = np.append(matdata_coo, mat_str[:-1].astype(np.float))
                        if extra_line == 2:
                            seekback = len(mat_str[-1].encode('utf-8')) + len(mat_str[-2].encode('utf-8'))
                            if last_char == '\n':
                                nlc = 2
                            else:
                                nlc = 1
                            matdata_coo = np.append(matdata_coo, mat_str[:-2].astype(np.float))
                        mmfile.seek(mmfile.tell()-(seekback+nlc))
                    else:
                        matdata_coo = np.append(matdata_coo, mat_str.astype(np.float))
                    mat_str = None
                    T = mmfile.read(chunksize)
                    readlength = len(T)
                    last_char = T[:-extra_line]
                print("")
                    #matdata_str = np.array(T)
                    #matdata_str = matdata_str.split()
                    #print(matdata_str)
                    #for line in T:
                    #    line = line.strip()
                    #    line = line.split()
                    #    line_np = np.array(line)
                    #    matdata_coo = np.append(matdata_coo, line_np.astype(np.float))
                    #T = T.strip()
                    #T = T.split()
                    #print(matdata_coo)
                    #matdata_str = np.asarray(T)
                    #matdata_coo = matdata_str.astype(np.float)
                    #print(matdata_str)
                    #return
                    #tmp_matdata_coo = matdata_str.astype(np.float)
                    #matdata_coo = np.append(matdata_coo, matdata_str.astype(np.float))
                    #matdata_coo = np.append(matdata_coo, tmp_matdata_coo)
                    #T = mmfile.readlines(chunksize)
                #matdata_coo = np.array([]) 
                #for L in mmfile:
                #    L = L.split()
                #    L = np.array([L])
                #    t = L.astype(np.float)
                #    matdata_coo = np.append(matdata_coo, t)
                print('Expected: {}, Calculated: {}'.format(3*entries, len(matdata_coo)))
                if len(matdata_coo) != (3 * entries):
                    print('Data file does not contain expected amount of data.', \
                    'Check that number of data lines matches nonzero count.')
                    return
                T = None
                mat_str = None
                matdata_coo = matdata_coo.reshape(entries,3)
                # Subtract 1 for 0 indexing
                matdata_coo[:,0] = [x - 1 for x in matdata_coo[:,0].astype(int)]
                matdata_coo[:,1] = [x - 1 for x in matdata_coo[:,1].astype(int)]
                # Create sparse matrix
                if mat_type == 'csc':
                    matdata_csc = csc_matrix((matdata_coo[:, 2], (matdata_coo[:,0], matdata_coo[:,1])), shape=(rows, cols))
                    matdata_coo = None
                    return [matdata_csc]
                elif mat_type == 'csr':
                    matdata_csr = csr_matrix((matdata_coo[:, 2], (matdata_coo[:,0], matdata_coo[:,1])), shape=(rows, cols))
                    matdata_coo = None
                    return [matdata_csr]                
                

            elif field == 'complex':
                pass
            elif field == 'pattern':
               pass
        elif rep == 'pattern':
            pass

        if symm == 'symmetric':
            pass
        elif  symm == 'hermitian':
            pass
        elif symm == 'skew-symmetric':
            pass









