'''
Created on 21.04.2017

@author: khamidova
'''
import numpy as np
import subprocess
import os
import metrics


template = """NAME: {name}
TYPE: TSP
COMMENT: {name}
DIMENSION: {n_cities}
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: LOWER_ROW
EDGE_WEIGHT_SECTION
{matrix_s}
{edges_s}EOF"""

class TSPSolverNotFound(IOError):
    pass

def matrixToString(matrix):
    matrix_s = ""
    n_cities = len(matrix)

    #check if distance more than 10000
    if matrix[0,1]<100000:
        #do not multiply by 1000
        matrix*=10
    
    matrix=matrix.astype(np.int64)
    matrix_str=matrix.astype(str)
    matrix_s=''
    for j in range(0,n_cities):
        matrix_s += ' '.join([matrix_str[i,j] for i in xrange(j)])
        matrix_s +='\n'
    return matrix_s

def dumpMatrix(matrix, neighbours, name="tspmean"):
    n_cities = len(matrix)
    matrix_s=matrixToString(matrix)

    if neighbours:
        fixed_edges_s= 'FIXED_EDGES_SECTION \n'+str(n_cities)+' 1\n'
    else:
        fixed_edges_s=''
    #print matrix_s
    #print template.format(**{'name': name,
    #                          'n_cities': n_cities,
    #                          'matrix_s': matrix_s})
    return template.format(**{'name': name,
                              'n_cities': n_cities,
                              'matrix_s': matrix_s,
                              'edges_s':fixed_edges_s})

def createLKHpar(tsp_path, runs=4):
    prefix, _ = os.path.splitext(tsp_path)
    par_path = prefix + ".par"
    out_path = prefix + ".out"
    par = """PROBLEM_FILE = {}
RUNS = {}
TOUR_FILE =  {}
TIME_LIMIT = 2500
PRECISION = 1

""".format(tsp_path, runs, out_path)
#
    with open(par_path, 'w') as dest:
        dest.write(par)

    return par_path, out_path

def runLKH(tsp_path, start=None):
    bdir = os.path.dirname(tsp_path)
    os.chdir(bdir)

    #LKH = os.environ.get('LKH', 'LKH')
    #print source_lib
    import config
    LKH=config.get_lkh_lib()
    
    par_path, out_path = createLKHpar(tsp_path)
    try:
        output = subprocess.check_output([LKH, par_path], shell=False)
        #print output
    except OSError as exc:
        #print str(exc)
        if "No such file or directory" in str(exc):
            raise TSPSolverNotFound(
                "{0} is not found on your path or is not executable".format(LKH))
        elif "The system cannot find the file specified" in str(exc):
            raise TSPSolverNotFound(
                "{0} is not found on your path or is not executable".format(LKH))
            

    meta = []
    raw = []
    solution = None
    with open(out_path) as src:
        header = True
        for line in src:
            if header:
                if line.startswith("COMMENT : Length ="):
                    solution = int(line.split(" ")[-1])
                if line.startswith("TOUR_SECTION"):
                    header = False
                    continue
                meta.append(line)
                continue

            else:
                if line.startswith("-1"):
                    break
                else:
                    raw.append(int(line.strip()) - 1)  # correct to zero-indexed

    metadata = "".join(meta)
        
    if start:
        # rotate to the beginning of the route
        while raw[0] != start:
            raw = raw[1:] + raw[:1]

    return {'tour': raw,
            'solution': solution,
            'metadata': metadata}

def solveTSPLKH(data,distance='euclidian',debug=False):
    
    dist_matrix=metrics.distanceMatrix(data,distance=distance)
    matrix_s=dumpMatrix(dist_matrix,False)

    
    if debug:
        print matrix_s
        
    thisdir = os.path.abspath(os.path.dirname(__file__))
    #print 'Run LKH'
    outf = os.path.join(thisdir, "test.tsp")
    with open(outf, 'w') as dest:
        dest.write(matrix_s)
    tsp_result = runLKH(outf,0)
    #order_list=tsp_result['tour']
    
    #resulting tour
    return tsp_result

def solveTSPLKHfile(filename):

    thisdir = os.path.abspath(os.path.dirname(__file__))
    
    outf = os.path.join(thisdir, filename)

    tsp_result = runLKH(outf,0)

    return tsp_result