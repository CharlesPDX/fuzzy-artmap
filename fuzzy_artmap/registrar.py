from tornado.tcpserver import TCPServer
from tornado.iostream import StreamClosedError
import tornado.ioloop

class RegistrarServer(TCPServer):
    def __init__(self, ssl_options = None, max_buffer_size = None, read_chunk_size = None) -> None:
        self.clients = set()
        super().__init__(ssl_options, max_buffer_size, read_chunk_size)

    async def handle_stream(self, stream, address):
        buffer_size = 4096
        total_data = bytearray()
        while True:
            try:
                # data = await stream.read_until(b"\n")
                # await stream.write(data)
                data = await stream.read_bytes(buffer_size, True)
                total_data.extend(data)
                if not data or len(data) < buffer_size:
                    await self.handle_data(total_data, stream)
                    total_data.clear()
            except StreamClosedError:
                print("connection closed")
                break
    
    async def handle_data(self, data, stream):
        if data[0] == 114: # "r"
            client_adress = data[1:].decode("utf-8").rstrip()
            self.clients.add(client_adress)
            print(f"Registering client: {client_adress}")
            await stream.write(bytes("registered\n", "utf-8"))
        elif data[0] == 103: # "g"
            print("received list clients request")
            send_clients = bytes(",".join(self.clients)+"|||", "utf-8")
            await stream.write(send_clients)
        else:
            print(f"unknown option: {data[0]}")

if __name__ == "__main__":
    server = RegistrarServer()
    print('Starting the server...')
    server.listen(8786)
    tornado.ioloop.IOLoop.current().start()
    print('Server has shut down.')
