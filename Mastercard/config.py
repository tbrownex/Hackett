def getClient(client):
    dataloc = {}
    dataloc["MC"]       = "/home/tbrownex/data/Hackett/Mastercard/"
    dataloc["JLP"]      = "/home/tbrownex/data/Hackett/JLP/"
    dataloc["OutFront"] = "/home/tbrownex/data/Hackett/OutFront/"
    return dataloc[client]