'''
Created on 17.03.2017

@author: khamidova
'''

from anytree import Node, RenderTree

def printTree(root_node):
    for pre, fill, node in RenderTree(root_node):
        print("%s%s" % (pre, node.name))

def addNode(value,parent_node=None,index=None):
    if parent_node==None:
        return Node(value,value=value, index=index)
    else:
        return Node(value,parent_node,value=value,index=index)
    
def changeParent(node,parent_node):
    node.parent=parent_node
    
def deleteNode(node,with_control=True):
    #check if there are children
    
    if with_control and len(node.children)>0 :
        print node.children
        print 'Cannot delete the node', node
        return -1
    
    #if not - change parent to None
    node.parent=None
    
def makeOlder(node,level):
    if level>0:
        anchestors=node.anchestors
        node_level=len(anchestors)
        if node_level>level:
            node.parent=anchestors[node_level-level-1]
        else:
            node.parent=node.root
    
def nodesToList(nodes_list):
    result=[]
    for current_node in nodes_list:  
        result.append(current_node.value)  
    return result

def nodesToIndexList(nodes_list):
    result=[]
    for current_node in nodes_list:  
        result.append(current_node.index)  
    return result