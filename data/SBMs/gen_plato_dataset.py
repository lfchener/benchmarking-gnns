import os
import pickle
import dgl
import numpy as np
from SBMs import SBMsDataset

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def main():
    name = 'SBM_PATTERN'
    dataset = SBMsDataset(name)
    
    # print("[I] Loading dataset %s..." % (name))
    # with open(name+'.pkl',"rb") as f:
    #     f = pickle.load(f)
    #     train = f[0]
    #     val = f[1]
    #     test = f[2]
    # print('train, test, val sizes :',len(train),len(test),len(val))
    # print("[I] Finished loading.")
    train = dataset.train
    test = dataset.test
    val = dataset.val

    train_nodes = []
    val_nodes = []
    test_nodes = []
    nodes = []
    node_fea = []
    edge_fea = []
    edges = []
    labels = []
    idx = add_node(train, 0, nodes, train_nodes, edges, node_fea, edge_fea, labels)
    idx = add_node(test, idx, nodes, test_nodes, edges, node_fea, edge_fea, labels)
    idx = add_node(val, idx, nodes, val_nodes, edges, node_fea, edge_fea, labels)
    print("train nodes:", len(train_nodes))
    print("val_nodes:", len(val_nodes))
    print("test_nodes:", len(test_nodes))
    print("nodes:", len(nodes))
    print("node_fea:", len(node_fea))
    print("edge_fea:", len(edge_fea))
    print("edges:", len(edges))
    print("labels:", len(labels))
    save_path = "SBMPattern"
    np.savetxt(os.path.join(save_path, 'train_nodes'), np.array(train_nodes), fmt='%d', delimiter=" ")
    np.savetxt(os.path.join(save_path, 'test_nodes'), np.array(test_nodes), fmt='%d', delimiter=" ")
    np.savetxt(os.path.join(save_path, 'val_nodes'), np.array(val_nodes), fmt='%d', delimiter=" ")
    np.savetxt(os.path.join(save_path, 'nodes'), np.array(nodes), fmt='%d', delimiter=" ")
    np.savetxt(os.path.join(save_path, 'edges'), np.array(edges), fmt='%d', delimiter=" ")
    np.savetxt(os.path.join(save_path, 'node_fea'), np.array(node_fea), fmt='%s', delimiter=" ")
    np.savetxt(os.path.join(save_path, 'edge_fea'), np.array(edge_fea), fmt='%s', delimiter=" ")
    np.savetxt(os.path.join(save_path, 'labels'), np.array(labels), fmt='%d', delimiter=" ")

def add_node(dataset, idx, nodes, part_nodes, edges, node_fea, edge_fea, labels):
    for graph, label in dataset:
        src, dst = graph.edges()
        src = src.data.numpy() + idx
        dst = dst.data.numpy() + idx

        # save edges
        edges.extend(np.stack([src, dst], axis=1).tolist())

        node_features = graph.ndata['feat'].data.numpy()
        for i in range(node_features.shape[0]):
            output = str(i + idx) + ' ' + str(node_features[i])
            node_fea.append(output)
        
        edge_features = graph.edata['feat'].data.numpy()
        for s, d, fea in zip(src, dst, edge_features):
            output = str(s) + ' ' + str(d) + ' '
            output += ','.join(map(lambda x: "{:.6f}".format(x), list(fea)))
            edge_fea.append(output)

        label = label.data.numpy()
        for i in range(label.shape[0]):
            labels.append([i+idx, label[i]])

        graph_node = list(range(idx, graph.num_nodes()+idx))
        nodes.extend(graph_node)
        part_nodes.extend(graph_node)
        idx += graph.num_nodes()
    return idx

if __name__ == '__main__':
    main()



    