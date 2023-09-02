package interFace;

import java.io.Serializable;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.*;

public class Nodes extends UnicastRemoteObject implements No{
    private Map<String, String> data;//键值存储

    public Nodes() throws RemoteException {
        super();
        this.data = new HashMap<>();//无序
    }
    public void put(String key, String value){
        data.put(key, value);
        System.out.println("put: " + key + " and " + value);
        System.out.println();
    }
    public String get(String key){
        return data.getOrDefault(key, null);
    }

    public void remove(String key){
        data.remove(key);
    }

    public List<String> outputall(){
        List<String> output = new ArrayList<String>();
        for (String key:data.keySet()){
            output.add("key=" + key + " value:" + data.get(key));
        }

        return output;
    }

    public Map<String, String> getAlldata(){
        return data;
    }

}
