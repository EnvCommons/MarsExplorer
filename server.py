from openreward.environments import Server

from marsexplorer import MarsExplorerEnvironment

if __name__ == "__main__":
    server = Server([MarsExplorerEnvironment])
    server.run()
