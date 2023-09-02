package interFace;

import java.rmi.NotBoundException;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public interface Store extends Remote {
    void addNode(String node, int port) throws RemoteException, NotBoundException, NoSuchAlgorithmException;
    int removeNode(String node) throws RemoteException, NoSuchAlgorithmException;
    String getNode(String key) throws RemoteException, NoSuchAlgorithmException;
    int getHash(String key) throws RemoteException, NoSuchAlgorithmException;
    List<String> outputAllNodes()throws RemoteException;
    List<String> outputAlldata(String node) throws NoSuchAlgorithmException, RemoteException;
    String put(String data_key, String data_value) throws RemoteException, NoSuchAlgorithmException;
    int remove(String data_key) throws RemoteException, NoSuchAlgorithmException;
    String get(String data_key) throws RemoteException, NoSuchAlgorithmException;
}
