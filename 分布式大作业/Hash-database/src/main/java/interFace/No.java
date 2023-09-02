package interFace;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;
import java.util.Map;

public interface No extends Remote {
    void put(String key, String value) throws RemoteException;
    String get(String key) throws RemoteException;
    void remove(String key) throws RemoteException;
    List<String> outputall() throws RemoteException;
    Map<String, String> getAlldata() throws RemoteException;
}
