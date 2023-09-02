package interFace;

import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.*;

import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class ConsistentHashingKVStore extends UnicastRemoteObject implements Store {
    private List<String> nodes;//节点列表
    List<No> node_list = new ArrayList<No>();//节点使用方法列表
    private TreeMap<Integer, String> ring;//一致性哈希环

    List<Integer> Init_PORT = new ArrayList<Integer>(){
        {
            add(1091);
            add(1092);
            add(1093);
            add(1094);
        }
    };

    public ConsistentHashingKVStore(List<String> initnodes) throws RemoteException, NotBoundException, NoSuchAlgorithmException {
        super();
        nodes = initnodes;
        ring = new TreeMap<>();

        // 初始化一致性哈希环(设置初始节点)
        int i=0;
        for (String node : nodes) {
            int hash = getHash(node);
            ring.put(hash, node);

            //注册
            Registry registry = LocateRegistry.getRegistry(Init_PORT.get(i));
            node_list.add((No) registry.lookup(node));
            i++;
        }
    }

    public void addNode(String node, int port) throws RemoteException, NotBoundException, NoSuchAlgorithmException {
        int hash = getHash(node);

        //分担前一个节点数据
        int index = nodes.indexOf(getNode(node));

        ring.put(hash, node);
        nodes.add(node);

        No n = node_list.get(index);
        Map<String, String> alldata;
        alldata = n.getAlldata();

        //注册
        Registry registry = LocateRegistry.getRegistry(port);
        node_list.add((No) registry.lookup(node));

        for(Map.Entry<String, String> entry: alldata.entrySet()) {
            remove(entry.getKey());
            put(entry.getKey(), entry.getValue());
        }

    }

    public int removeNode(String node) throws NoSuchAlgorithmException, RemoteException {
        int hash = getHash(node);
        ring.remove(hash, node);

        int out = 0;

        int index = nodes.indexOf(node);

        if(index >= 0){
            //节点数据转移
            No n = node_list.get(index);
            Map<String, String> alldata = new HashMap<>();
            alldata = n.getAlldata();
            for(Map.Entry<String, String> entry: alldata.entrySet()) {
                put(entry.getKey(), entry.getValue());
            }

            node_list.remove(index);
            nodes.remove(index);
            out = 1;
        }



        return out;
    }

    public String getNode(String key) throws NoSuchAlgorithmException {
        int hash = getHash(key);

        Map.Entry<Integer, String> entry = ring.ceilingEntry(hash);//返回一个具有大于或等于key的最小键的条目，否则为null
        if(entry == null){
            entry = ring.firstEntry();//方法用于返回与此映射中的最小键关联的键值映射，如果映射为空，则返回 null(key==value)，这里返回第一个映射
        }

        return entry.getValue();
    }

    public int getHash(String key) throws NoSuchAlgorithmException {
        //int hash = (511) & (key.hashCode()^(key.hashCode()>>>16));数据不分散

        //MD5哈希值
        MessageDigest md = MessageDigest.getInstance("MD5");
        byte[] hashBytes = md.digest(key.getBytes());
        BigInteger hashInt = new BigInteger(1, hashBytes);

        //使用mod()方法将大整数转化为0-1000内
        int smallHash = hashInt.mod(BigInteger.valueOf(1000)).intValue();


        return smallHash;
    }

    public List<String> outputAllNodes(){
        List<String> output = new ArrayList<String>();

        for(Map.Entry<Integer, String> r : ring.entrySet()){
            output.add("hash-value:" + r.getKey() + "     ip:" + r.getValue());
        }

        return output;
    }
    public List<String> outputAlldata(String node) throws NoSuchAlgorithmException, RemoteException {
        int hash = getHash(node);

        int index = nodes.indexOf(node);
        if (index<0)
            return null;

        //节点数据转移
        No n = node_list.get(index);
        Map<String, String> alldata = new HashMap<>();
        alldata = n.getAlldata();

       List<String> output = new ArrayList<String>();
        for(Map.Entry<String, String> r : alldata.entrySet()){
            output.add("key:" + r.getKey() + " value:" + r.getValue());
        }

        return output;
    }

    public String put(String data_key, String data_value) throws RemoteException, NoSuchAlgorithmException {
        String node = getNode(data_key);
        int index = nodes.indexOf(node);

        //输入数据
        No n = node_list.get(index);
        n.put(data_key, data_value);

        return node;
    }

    public int remove(String data_key) throws RemoteException, NoSuchAlgorithmException {
        //先获取相应节点
        String node = getNode(data_key);
        int index = nodes.indexOf(node);

        //对节点进行处理
        No n = node_list.get(index);
        String value = n.get(data_key);

        int out = 0;

        if(value != null)
        {
            n.remove(data_key);
            out = 1;
        }
        return out;
    }

    public String get(String data_key) throws RemoteException, NoSuchAlgorithmException {
        //先获取相应节点
        String node = getNode(data_key);
        int index = nodes.indexOf(node);

        //对节点进行处理
        No n = node_list.get(index);
        String value = n.get(data_key);

        String out;

        if(value == null)
            out = null;
        else {

            out = "数据values: " + value;
            out += "   数据存储节点: " + node;
        }
        return out;
    }
}
