import interFace.*;

import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;
import java.util.List;

public class RegistryServer implements Re{
    public static  void main(String[] args) throws Exception {
        List<String> nodes = new ArrayList<String>() {
            {
                add("rmi://127.0.0.2");
                add("rmi://127.0.0.3");
                add("rmi://127.0.0.4");
                add("rmi://127.0.0.5");
            }
        };

        //绑定节点（初始4个节点）
        for (int i = 0; i < 4; i++) {
            int port = 1090 + 1 + i;
            Registry re = LocateRegistry.createRegistry(port);
            String na = nodes.get(i);//服务名称
            No n = new Nodes();
            re.rebind(na, n);
            System.out.println("=====启动RMI节点服务成功，服务地址：" + na + "====");
        }

        String RMI_NAME = "rmi://127.0.0.1";//服务名称
        Registry registry = LocateRegistry.createRegistry(1090);
        Store store = new ConsistentHashingKVStore(nodes);
        registry.rebind(RMI_NAME, store);
        System.out.println("=====启动RMI数据库服务成功，服务地址：" + RMI_NAME + "====");
    }

    //注册新节点
    public void RegisterNewNode(String ip, int port) throws RemoteException {
        Registry re = LocateRegistry.createRegistry(port);
        No n = new Nodes();
        re.rebind(ip, n);
        System.out.println("=====启动RMI节点服务成功，服务地址：" + ip + "====");
    }
}