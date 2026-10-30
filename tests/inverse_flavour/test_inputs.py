import math
from src.inverse_flavour.inputs import Id_device, yukawas_to_device, CKM_device, init_state_sm
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tests of Id_device
    print("Tests of Id_device:")
    Id_test = Id_device(device=device)
    print("[Check 1]: Identity on correct device")
    assert Id_test.shape == (3,3)
    assert str(Id_test.device) == str(device)
    print("Passed")

    #Tests of yukawa_to_device
    print("Tests of yukawa_to_device:")
    yu_test, yd_test = yukawas_to_device(device=device)
    print("Check[1]: Up Yukawa diagonal")
    assert (yu_test[1] == yu_test[2]== yu_test[3] ==yu_test[5] ==
             yu_test[6] ==yu_test[7]==0
    )
    print("Passed")
    
    print("Check[2]: Down Yukawa diagonals are in correct order (light to heavy)")
    assert torch.abs(yd_test[8]) > torch.abs(yd_test[4])
    assert torch.abs(yd_test[4]) > torch.abs(yd_test[0])
    print("Passed")






if __name__ == "__main__":
    main()