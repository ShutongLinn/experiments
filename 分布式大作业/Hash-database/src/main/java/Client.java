import interFace.No;
import interFace.Store;

import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.rmi.AccessException;
import java.rmi.NotBoundException;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;


public class Client {
    public static void main(String[] args){
        RegistryServer registryServer = new RegistryServer();

        int num = 1095;

        try {
            //通过服务端绑定的名称(RMI_NAME)从注册表中获取对象(lookup()方法)
            //获取数据库方法
            Registry re = LocateRegistry.getRegistry(1090);
            Store kvStore = (Store) re.lookup("rmi://127.0.0.1");


            System.out.println("====>KVStore has already stored 4 nodes<====");
            System.out.println();

            while (true){
                System.out.println();
                System.out.println("====>1.增加数据<========>2.删除数据<========>3.查询数据<====");
                System.out.println("====>4.增加节点<========>5.删除节点<=======>6.输出所有节点<===");
                System.out.println("====>7.输出节点数据<====>8.退出系统<========>");
                System.out.println("====>请输入相关操作<====");
                System.out.println();

                Scanner input = new Scanner(System.in);
                int in = input.nextInt();
                switch (in){
                    case 1: {
                        System.out.println("====>请输入所增加数据的key和value，以空格分开<====");
                        String data_key = input.next();
                        String data_value = input.next();

                        //通过kvStore把数据进行put操作
                        String out = kvStore.put(data_key, data_value);
                        System.out.println("数据哈希值: "+ kvStore.getHash(data_key));
                        System.out.println("数据存储节点地址：" + out);
                    }break;

                    case 2:{
                        System.out.println("====>请输入需要删除数据的key<====");
                        String data_key = input.next();

                        int out = kvStore.remove(data_key);

                        if(out == 0)
                            System.out.println("删除失败！没有存储该数据!");
                        else
                            System.out.println("删除成功!");
                    }break;

                    case 3:{
                        System.out.println("====>请输入所查询数据的key<====");
                        String data_key = input.next();

                        String out = kvStore.get(data_key);
                        if(out != null)
                        {
                            System.out.println(out);
                        }else
                            System.out.println("没有查询到相应数据");
                    }break;

                    case 4:{
                        System.out.println("====>请输入所增加节点node的ip地址<====");
                        String node_key = input.next();

                        //通过接口在数据库和服务器注册
                        registryServer.RegisterNewNode(node_key, num);
                        kvStore.addNode(node_key, num);
                        num ++;

                        System.out.println("增加节点成功！");
                        System.out.println("节点哈希值: "+ kvStore.getHash(node_key));
                    }break;

                    case 5:{
                        System.out.println("====>请输入需要删除节点node的ip地址<====");
                        String node_key = input.next();

                        int out = kvStore.removeNode(node_key);
                        if(out == 1)
                            System.out.println("删除节点成功！");
                        else
                            System.out.println("删除失败！未查询到该节点");
                    }break;

                    case 6:{
                        List<String> out = new ArrayList<String>();
                        out = kvStore.outputAllNodes();

                        for(String i : out)
                        {
                            System.out.println(i);
                        }
                    }break;
                    case 7:{
                        System.out.println("====>请输入需要查询节点node的ip地址<====");
                        String node_key = input.next();
                        List<String> out;
                        out = kvStore.outputAlldata(node_key);
                        if(out == null)
                            System.out.println("查询错误！未查询到该节点！");

                        System.out.println(out);

                    }break;

                    case 8:{
                        System.exit(0);
                    }break;

                }
            }

        } catch (AccessException e) {
            throw new RuntimeException(e);
        } catch (NotBoundException e) {
            throw new RuntimeException(e);
        } catch (RemoteException e) {
            throw new RuntimeException(e);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }
}
