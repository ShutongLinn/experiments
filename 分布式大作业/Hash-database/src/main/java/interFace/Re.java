package interFace;

import java.rmi.RemoteException;

public interface Re {
    void RegisterNewNode(String ip, int port) throws RemoteException;
}
