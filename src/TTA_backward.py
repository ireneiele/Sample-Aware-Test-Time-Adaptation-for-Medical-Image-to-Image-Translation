from models.adaptor_3 import DTTAnorm
from models.adaptor_3 import ANet
import torch
import torch.nn as nn
import torch.optim as optim
from models.UNet import UNet
from models.Autoencoder_model import AENet
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
import pandas as pd
import os
from util import html
from util.visualizer import save_images
import torch
from models import find_model_using_name
from util.visualizer import calcola_mse, calculate_psnr, calculate_ssim
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time

# soppriamo il warning
import warnings
warnings.filterwarnings("ignore")

# from util.functional import normalize
# from util.Variable import Variable


# questo script serve per la fase di TTA, in cui vengono trainati gli Adaptor e freezati Task network e Autoencoder

torch.manual_seed(0)  # garantisce la riproducibilità
torch.cuda.manual_seed(0)
np.random.seed(0)
#torch.use_deterministic_algorithms(True)
from options.base_options import BaseOptions
from tqdm import tqdm

thresholdes = {'denoising': {80: 0.0024, 85: 0.0028, 90: 0.0033, 95: 0.0064, 98: 0.011},
               'brats': {80: 0.0009, 85: 0.0010, 90: 0.0010, 95: 0.0012, 98: 0.0014}}

def compute_tnet_dim(opt):
    layers_to_dim = {'input': 1, 'first_conv': 64, 'second_conv': 128, 'third_conv': 256, 'resnet_block_1': 256,
                     'resnet_block_2': 256, 'resnet_block_3': 256, 'resnet_block_4': 256, 'final_output': 1}
    tnet_dim = []
    for layer in opt.return_layers:
        tnet_dim.append(layers_to_dim[layer])
    opt.tnet_dim = tnet_dim


def l2_reg_ortho(model, lambda_l2=1e-4):
    l2_loss = torch.tensor(0.0, device='cuda')
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2  # norma L2 dei parametri
    return lambda_l2 * l2_loss


def backward_step(opt, all_comb, batch, model_type='AENet'):
    orthw = opt.__dict__.get('orthw', 1)
    n = len(return_layers[1:-1])  # i ricostruttori escluso il primo e l'ultimo
    loss_config_output = pd.DataFrame(columns=['config', 'loss_output', 'loss_tot', 'PSNR'])
    # adesso eliminiamo una alla volta un elementp da all_comb
    for i in range(len(all_comb)):
        comb_to_print = all_comb[:i] + all_comb[i + 1:]  # elimino l'elemento i
        chosen_comb = list(comb_to_print)  # prendo una combinazione
        chosen_comb = sorted(chosen_comb)  # la ordino
        comb_to_print = chosen_comb
        print('formato', chosen_comb)
        chosen_comb = [0] + [x + 1 for x in chosen_comb] + [n + 1]
        chosen_comb = [return_layers[i] for i in chosen_comb]
        opt.return_layers = chosen_comb
        compute_tnet_dim(opt)

        if opt.rec_model == 'AENet':
            AE = AENet(opt)
            for i in range(len(opt.return_layers)):
                name = opt.return_layers[i]
                elem = opt.tnet_dim[i]
                if opt.task == 'denoising':
                    load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints2/train_AE/epoch99/AE_{name}_99.pt'

                elif opt.task == 'brats':
                    load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints3/train_all_AE/epoch99/AE_{name}_99.pt'

                # cesm
                # load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints4/train_all_AE_cesm/epoch99/AE_{name}_99.pt'
                
                # IXI
                # load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints3/train_all_AE_IXI/epoch99/AE_{name}_99.pt'



                state_dict = torch.load(load_path_weights, map_location=str(
                    AE.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi

                AE.AENet[i].load_state_dict(state_dict)  # così carico il dizionario nel modello
                AE.set_requires_grad(AE.AENet[i], False)
            rec_models = AE.AENet

        adaptors.reset(default=False)  # resetta per ogni campione i pesi

        # pretraining degli Adaptor
        for epoch in range(opt.sepochs):
            m_loss = 0
            outputs = adaptors(batch, task_model)
            loss = adaptors.ALoss(outputs['input'], task_model.real_A)  # calcola la loss tra l'immagine di input e adattata
            org_loss = orthw * l2_reg_ortho(adaptors.conv)  # orthogonal regularization
            loss += org_loss
            m_loss += loss
            adaptors.optimizer_ANet.zero_grad()
            loss.backward()
            adaptors.optimizer_ANet.step()
            # print(f'loss pre training epoch {epoch}: {m_loss}')

        # start_time = time.time()
        prev_loss = float('inf')  # loss precedente, diamo un valore molto grande
        loss_tot = []
        loss_output = []

        for epoch in range(opt.tepochs):  # numero di iterazioni per ogni campione
            outputs = adaptors(batch, task_model)

            loss = 0  # somma delle 4 loss (una per ogni adaptor)
            print('---------------------------------')

            for i in range(len(rec_models)):
                index = opt.return_layers[i]  # prendo l'indice del layer (chiave del dizionario outputs)
                side_out = outputs[index]  # side_out è l'output del task network
                level_loss = 0  # loss del singolo reconstruction model
                """Keep this part the same with opt_AENet"""
                if model_type == 'AENet':
                    if len(AE.AENetMatch[i]) == 2:  # concatenate features from the same level
                        side_out_cat = torch.cat([side_out[0], side_out[1]], dim=1)
                    else:
                        # use seperate features
                        side_out_cat = side_out  # TODO: preso da train_AE, prima era  side_out_cat = side_out[self.AENetMatch[_][0]]

                    ae_out = AE.AENet[i](side_out_cat,
                                         side_out=False)  # Il modello di ricostruzione riceve l' input, Questo produce l'immagine ricostruita.
                    level_loss = weights[i] * AE.AELoss(ae_out,
                                                        side_out_cat)  # La loss è calcolata confrontando l'immagine ricostruita (ae_out) con l'immagine adattata originale (side_out_cat).

                # if i != len(rec_models) - 1:
                #    loss += level_loss # somma delle loss dei vari livelli
                print(f'loss {i} epoch {epoch}: {level_loss}')
                loss += level_loss  # somma delle loss dei vari livelli

            loss_output.append(level_loss.data.item())  # loss dell'output
            org_loss = orthw * l2_reg_ortho(adaptors.conv)  # orthogonal regularization TODO: modificta io
            loss += org_loss
            loss_tot.append(loss.data.item())

            adaptors.optimizer_ANet.zero_grad()
            loss.backward()  # faccio backward della somma delle loss (include sia la perdita derivante dai
            # modelli di ricostruzione che quella derivante dalla regolarizzazione ortogonale)
            adaptors.optimizer_ANet.step()
            adaptors.save_networks(str(epoch))

            # Early stopping
            if prev_loss < loss:
                break
            else:
                prev_loss = loss

        min_index = loss_output.index(min(loss_output))  # indice dell'epoca che ha prodotto la perdita minima
        print (comb_to_print)
        used_comb = [str(x + 1) for x in comb_to_print]  # prendo la combinazione usata
        used_comb = '_'.join(used_comb)  # la trasformo in stringa
        adaptors.load_networks(str(min_index))  # carico i pesi del modello con la loss minore
        adaptors.save_networks(used_comb, config=True)  # salvo i pesi del modello con la combinazione usata
        # inseriamo in loss_config_output la riga di loss_output a cui corrisponde il minimo
        print(used_comb)

        # OUTPUT
        with torch.no_grad():
            outputs = adaptors(batch, task_model)  # passo forward degli adaptor
        real_B = task_model.real_B  # campione reale
        fake_B = outputs[opt.return_layers[-1]]  # campione adattato

        visuals_output = OrderedDict()
        visuals_output['real_B'] = real_B
        visuals_output['fake_B'] = fake_B

        psnr_score = calculate_psnr(visuals_output)
        loss_config_output = loss_config_output.append(
            {'config': used_comb, 'loss_output': min(loss_output),
             'loss_tot': loss_tot[min_index] / len(used_comb), 'PSNR': psnr_score}, ignore_index=True)

    return loss_config_output # contiene le loss delle configurazioni provate dalla funzione

    # aggiunge a loss_config la loss di ogni configurazione

def opt_ANet(adaptors, opt, task_model, rec_models=None, stable=False, model_type='AENet', AE=None, return_layers=None):
    stop_plot = 0
    n = len(return_layers[1:-1])  # i ricostruttori escluso il primo e l'ultimo
    all_comb = [0,1,2,3,4,5,6]
    # rec_loss = 0.0063 #task denoising
    thr = thresholdes[opt.task][opt.confidence]

    mse_o = []
    psnr_o = []
    ssim_list_o = []
    df_o = pd.DataFrame(columns=['img_name', 'config', 'MAE', 'PSNR', 'SSIM'])

    mse_task_model = []
    psnr_task_model = []
    ssim_list_task_model = []
    df = pd.DataFrame(columns=['img_name', 'MAE', 'PSNR', 'SSIM'])

    task_model.set_requires_grad([task_model.netG_A, task_model.netG_B, task_model.netD_A, task_model.netD_B], False)
    adaptors.set_requires_grad([adaptors.adpNet, adaptors.conv], True)  #
    task_model.eval()

    for subnets in rec_models:
        subnets.eval()
    adaptors.train()
    # weights = opt.__dict__.get('weights', [1]*len(AENet.AENet))
    orthw = opt.__dict__.get('orthw', 1)

    count_tta = 0
    count_tta_good = 0
    # stablize training by pre-train histogram manipulator (if needed)

    if os.path.exists(os.path.join(opt.results_dir, 'metrics', f'TTA_output.csv')):
        already_done_df = pd.read_csv(os.path.join(opt.results_dir, 'metrics', f'TTA_output.csv'))
    else:
        already_done_df = None

    for j, batch in tqdm(enumerate(dataset.dataloader)):  # arriva il campione di test
        print('CAMPIONE:', j)

        if already_done_df is not None:
            if str(batch['A_paths']) in already_done_df['img_name'].tolist():
                print(f"Skipping {batch['A_paths']} (already processed)")
                continue  # salta al prossimo
        all_comb = [0, 1, 2, 3, 4, 5, 6]
        # creiamo un csv vuoto chiamato loss_config_output che ha come colonne config e loss_output
        loss_config_output = pd.DataFrame(columns=['config', 'loss_output', 'loss_tot', 'PSNR'])
        task_model.set_input(batch)
        outputs = task_model.forward(return_layers=opt.return_layers)  # passo forward del task model

        visuals_task_model = task_model.get_current_visuals()
        img_path_task_model = task_model.get_image_paths()

        mae_score_task_model = calcola_mse(visuals_task_model)  # dovrebbe essere per ogni batch
        mse_task_model.append(mae_score_task_model)
        psnr_score_task_model = calculate_psnr(visuals_task_model)
        psnr_task_model.append(psnr_score_task_model)
        ssim_score_task_model = calculate_ssim(visuals_task_model)
        ssim_list_task_model.append(ssim_score_task_model)
        row = {'img_name': img_path_task_model, 'MAE': mae_score_task_model, 'PSNR': psnr_score_task_model,
               'SSIM': ssim_score_task_model}
        df_new_tm = pd.DataFrame([row])
        df = pd.concat([df, df_new_tm], ignore_index=True)

        real_to_plot = task_model.real_B.squeeze().cpu().numpy() # High-Dose
        no_tta_to_plot = task_model.fake_B.squeeze().cpu().numpy() # High-Dose No-TTA

        index = opt.return_layers[-1]  # prendo l'indice del layer (chiave del dizionario outputs)
        side_out = outputs[index]  # side_out è l'output del task network
        ae_out= AE.AENet[-1](side_out, side_out=False)  # uscita del modello di ricostruzione -> dominio B
        rec_loss = weights[-1] * AE.AELoss(ae_out, side_out)
        # arrotomdiamo la loss a 4 cifre decimali
        rec_loss = round(rec_loss.item(), 4)

        # appendiamo la rec_loss alla lista rec_loss

        """
        if opt.use_online and j < 100:
            thr_online.append(rec_loss)
        elif opt.use_online and j >= 100:
            thr_online.append(rec_loss)
            thr = np.percentile(thr_online, 95)
        """

        if rec_loss > thr:
            count_tta += 1
            start_time = time.time()
            # aggiungiamo tutte le combinazioni possibili ad una lista
            chosen_comb = list(all_comb)  # prendo una combinazione
            chosen_comb = sorted(chosen_comb)  # la ordino
            comb_to_print = chosen_comb
            chosen_comb = [0] + [x + 1 for x in chosen_comb] + [n + 1]
            chosen_comb = [return_layers[i] for i in chosen_comb]  # scelgo i layer corrispondenti
            opt.return_layers = chosen_comb
            compute_tnet_dim(opt)

            if opt.rec_model == 'AENet':
                AE = AENet(opt)
                for i in range(len(opt.return_layers)):
                    name = opt.return_layers[i]
                    elem = opt.tnet_dim[i]
                    
                    if opt.task == 'denoising':
                        load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints2/train_AE/epoch99/AE_{name}_99.pt'

                    elif opt.task == 'brats':
                        load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints3/train_all_AE/epoch99/AE_{name}_99.pt'

                    # cesm
                    # load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints4/train_all_AE_cesm/epoch99/AE_{name}_99.pt'
                    
                    # IXI
                    # load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints3/train_all_AE_IXI/epoch99/AE_{name}_99.pt'

                    state_dict = torch.load(load_path_weights, map_location=str(
                        AE.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi

                    AE.AENet[i].load_state_dict(state_dict)  # così carico il dizionario nel modello
                    AE.set_requires_grad(AE.AENet[i], False)
                rec_models = AE.AENet

            if opt.rec_model == 'pix2pix_rec':
                rec_models = []
                rec_model = find_model_using_name(opt.rec_model)
                for i in range(len(opt.return_layers)):
                    name = opt.return_layers[i]
                    elem = opt.tnet_dim[i]
                    opt.rec_input_nc = elem
                    opt.rec_output_nc = elem
                    if elem == 1:
                        rec_input = False
                    else:
                        rec_input = True

                    rm = rec_model(opt, rec_input)

                    load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints2/train_pix2pix/epoch99/pix2pixrec_{name}_G_99.pt'
                    state_dict = torch.load(load_path_weights, map_location=str(
                        rm.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi
                    if isinstance(rm.netG, torch.nn.DataParallel):
                        rm.netG.load_state_dict(state_dict)
                    else:
                        rm.netG.module.load_state_dict(state_dict)
                    rm.set_requires_grad([rm.netG, rm.netD], False)

                    rec_models.append(rm)
                AE = None

            adaptors.reset(default=True)  # resetta per ogni campione i pesi

            # pretraining degli Adaptor
            for epoch in range(opt.sepochs):
                m_loss = 0
                outputs = adaptors(batch, task_model)
                loss = adaptors.ALoss(outputs['input'], task_model.real_A)  # calcola la loss tra l'immagine di input e adattata
                org_loss = orthw * l2_reg_ortho(adaptors.conv)  # orthogonal regularization
                loss += org_loss
                m_loss += loss
                adaptors.optimizer_ANet.zero_grad()
                loss.backward()
                adaptors.optimizer_ANet.step()
                # print(f'loss pre training epoch {epoch}: {m_loss}')

            # start_time = time.time()
            prev_loss = float('inf')  # loss precedente, diamo un valore molto grande
            loss_tot = []
            loss_output = []

            for epoch in range(opt.tepochs):  # numero di iterazioni per ogni campione
                outputs = adaptors(batch, task_model)

                loss = 0  # somma delle 4 loss (una per ogni adaptor)
                print('---------------------------------')

                for i in range(len(rec_models)):
                    index = opt.return_layers[i]  # prendo l'indice del layer (chiave del dizionario outputs)
                    side_out = outputs[index]  # side_out è l'output del task network
                    level_loss = 0  # loss del singolo reconstruction model
                    """Keep this part the same with opt_AENet"""
                    if model_type == 'AENet':
                        if len(AE.AENetMatch[i]) == 2:  # concatenate features from the same level
                            side_out_cat = torch.cat([side_out[0], side_out[1]], dim=1)
                        else:
                            # use seperate features
                            side_out_cat = side_out  # TODO: preso da train_AE, prima era  side_out_cat = side_out[self.AENetMatch[_][0]]

                        ae_out = AE.AENet[i](side_out_cat,
                                                side_out=False)  # Il modello di ricostruzione riceve l' input, Questo produce l'immagine ricostruita.
                        level_loss = weights[i] * AE.AELoss(ae_out,side_out_cat)  # La loss è calcolata confrontando l'immagine ricostruita (ae_out) con l'immagine adattata originale (side_out_cat).

                    else:
                        if isinstance(side_out, tuple):  # se la lista interna ha due elementi
                            # concatenate features from the same level
                            side_out_cat = torch.cat([side_out[0], side_out[1]], dim=1)
                        else:
                            side_out_cat = side_out

                        rec_models[i].real_A = side_out_cat.to(rec_models[i].device)
                        rec_models[i].real_B = side_out_cat.to(rec_models[i].device)
                        rec_models[i].image_paths = task_model.image_paths

                        fake_B = rec_models[i].forward()
                        level_loss += adaptors.ALoss(fake_B,
                                                 side_out_cat) # calcola la loss tra l'immagine ricostruita da ricostruttore e quella adatatta

                    #if i != len(rec_models) - 1:
                    #    loss += level_loss # somma delle loss dei vari livelli
                    print(f'loss {i} epoch {epoch}: {level_loss}')
                    loss += level_loss  # somma delle loss dei vari livelli

                loss_output.append(level_loss.data.item()) # loss dell'output
                org_loss = orthw * l2_reg_ortho(adaptors.conv)  # orthogonal regularization TODO: modificta io
                loss += org_loss
                loss_tot.append(loss.data.item())


                adaptors.optimizer_ANet.zero_grad()
                loss.backward() # faccio backward della somma delle loss (include sia la perdita derivante dai
                # modelli di ricostruzione che quella derivante dalla regolarizzazione ortogonale)
                adaptors.optimizer_ANet.step()
                adaptors.save_networks(str(epoch))

                # Early stopping
                if prev_loss < loss:
                    break
                else:
                    prev_loss = loss

            min_index = loss_output.index(min(loss_output)) # indice dell'epoca che ha prodotto la perdita minima
            used_comb = [str(x + 1) for x in comb_to_print] # prendo la combinazione usata
            used_comb = '_'.join(used_comb) # la trasformo in stringa
            adaptors.load_networks(str(min_index)) # carico i pesi del modello con la loss minore
            adaptors.save_networks(used_comb, config=True) # salvo i pesi del modello con la combinazione usata
            # inseriamo in loss_config_output la riga di loss_output a cui corrisponde il minimo

            with torch.no_grad():
                outputs = adaptors(batch, task_model)  # passo forward degli adaptor
            real_B = task_model.real_B  # campione reale
            fake_B = outputs[opt.return_layers[-1]]  # campione adattato

            visuals_output = OrderedDict()
            visuals_output['real_B'] = real_B
            visuals_output['fake_B'] = fake_B
            psnr_score = calculate_psnr(visuals_output)

            loss_config_output = loss_config_output.append(
                {'config': used_comb, 'loss_output': loss_output[min_index],
                 'loss_tot': loss_tot[min_index] / len(used_comb), 'PSNR': psnr_score}, ignore_index=True)

            stop_condition = min(loss_output)
            new_loss = 0
            while new_loss < stop_condition: # finchè la loss non è minore della loss minima
                backward_csv = backward_step(opt, all_comb, batch)
                new_loss = backward_csv['loss_output'].min()
                all_comb = backward_csv[backward_csv['loss_output'] == new_loss]['config'].values[0].split('_') # prendo la combinazione con la loss minima
                # de all_comb è un solo elemento, new_loss è infinito
                if len(all_comb) == 1:
                    break
                # trasformo la stringa in una lista di interi
                all_comb = [int(x) - 1 for x in all_comb] #
                loss_config_output = pd.concat([loss_config_output, backward_csv], ignore_index=True) # aggiungo al csv

            # seleziono per il campione la migliore combinazione, cioè quella che minimizza la loss di output.
            min_loss = loss_config_output[opt.criteria].min() # prendo la loss minima

            # prendo la combinazione corrispondente alla loss minima
            used_comb = loss_config_output[loss_config_output[opt.criteria] == min_loss]['config'].values[0]
            if loss_config_output[loss_config_output[opt.criteria] == min_loss]['PSNR'].values[0] > psnr_score_task_model:
                count_tta_good += 1
            print('TOTALI TTA:', count_tta)
            print('GOOD TTA:', count_tta_good)

            if not os.path.exists(os.path.join(opt.results_dir, 'metrics')):
                os.makedirs(os.path.join(opt.results_dir, 'metrics'))
            with open(os.path.join(opt.results_dir, 'metrics', 'count.txt'), 'a') as f:
                f.write(f'TOTALI TTA: {count_tta}\n')
                f.write(f'GOOD TTA: {count_tta_good}\n')

            adaptors.load_networks(used_comb, config=True) # carico i pesi del modello con la combinazione usata
            # return layers deve diventare quella corretta
            chosen_comb = used_comb.split('_')  # divide la stringa in interi
            chosen_comb = [int(x) - 1 for x in chosen_comb] # int(x)- 1 effettua questa conversione, rendendo gli indici compatibili con il codice.
            chosen_comb = sorted(chosen_comb) # ordina
            chosen_comb = [0] + [x + 1 for x in chosen_comb] + [n + 1] # aggiunge il primo e l'ultimo layer
            chosen_comb = [return_layers[i] for i in chosen_comb]
            opt.return_layers = chosen_comb # scelgo i layer corrispondenti
            compute_tnet_dim(opt)

            with torch.no_grad():
                outputs = adaptors(batch, task_model)  # passo forward degli adaptor
            real_B = task_model.real_B  # campione reale
            fake_B = outputs[opt.return_layers[-1]]  # campione adattato
            img_path = task_model.get_image_paths()

            visuals_output = OrderedDict()
            visuals_output['real_B'] = real_B
            visuals_output['fake_B'] = fake_B

            # OUTPUT
            ssim_score = calculate_ssim(visuals_output)
            ssim_list_o.append(ssim_score)
            mae_score = calcola_mse(visuals_output)  # dovrebbe essere per ogni batch
            mse_o.append(mae_score)
            psnr_score = calculate_psnr(visuals_output)
            psnr_o.append(psnr_score)

            end_time = time.time()
            duration = end_time - start_time
            minutes = int(duration // 60)  # Minuti totali
            seconds = int(duration % 60)  # Secondi rimanenti

            print(f'Tempo impiegato per il campione{j}: {minutes} minuti e {seconds} secondi')
            # se non esiste la cartella la crea
            if not os.path.exists(os.path.join(opt.results_dir, 'metrics')):
                os.makedirs(os.path.join(opt.results_dir, 'metrics'))
            # salvo il tempo di esecuzione in un file txt
            with open(os.path.join(opt.results_dir, 'metrics', 'execution_time.txt'), 'a') as f:
                f.write(f'Tempo impiegato per il campione{j}: {minutes} minuti e {seconds} secondi\n')

            row = {'img_name': img_path, 'config': used_comb, 'MAE': mae_score, 'PSNR': psnr_score, 'SSIM': ssim_score,}
            df_new = pd.DataFrame([row])
            df_o = pd.concat([df_o, df_new], ignore_index=True)

            if not os.path.exists(os.path.join(opt.results_dir, 'metrics')):
                os.makedirs(os.path.join(opt.results_dir, 'metrics'))
            metrics_file_path = os.path.join(opt.results_dir, 'metrics', f'TTA_output.csv') # percorso del file csv
            # Aggiorna il file CSV progressivamente
            if not os.path.exists(metrics_file_path):
                df_o.to_csv(metrics_file_path, mode='w', index=False)
            else:
                df_new.to_csv(metrics_file_path, mode='a', index=False, header=False)

            if stop_plot < 50:
                if not os.path.exists(os.path.join(opt.results_dir, 'images', batch['A_name'][0])):
                    os.makedirs(os.path.join(opt.results_dir, 'images', batch['A_name'][0]))

                loss_config_output.to_csv(os.path.join(opt.results_dir, 'images', batch['A_name'][0], 'loss_config_output.csv'),
                                          index=False)
                
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(real_to_plot, cmap='gray')
                plt.subplot(1, 2, 2)
                plt.imshow(no_tta_to_plot, cmap='gray')
                plt.savefig(os.path.join(opt.results_dir, 'images', batch['A_name'][0], 'task_model.png'))

                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(real_to_plot, cmap='gray')
                plt.subplot(1, 2, 2)
                plt.imshow(fake_B.squeeze().cpu().numpy(), cmap='gray')
                plt.savefig(os.path.join(opt.results_dir, 'images', batch['A_name'][0], f'{used_comb}.png'))
                plt.show()

                stop_plot += 1

        else:
            ssim_score = ssim_score_task_model
            ssim_list_o.append(ssim_score)
            mae_score = mae_score_task_model  # dovrebbe essere per ogni batch
            mse_o.append(mae_score)
            psnr_score = psnr_score_task_model
            psnr_o.append(psnr_score)

            row = {'img_name': img_path_task_model, 'config': None, 'MAE': mae_score, 'PSNR': psnr_score, 'SSIM': ssim_score}
            df_new = pd.DataFrame([row])
            df_o = pd.concat([df_o, df_new], ignore_index=True)

            if not os.path.exists(os.path.join(opt.results_dir, 'metrics')):
                os.makedirs(os.path.join(opt.results_dir, 'metrics'))
            metrics_file_path = os.path.join(opt.results_dir, 'metrics', f'TTA_output.csv')  # percorso del file csv
            # Aggiorna il file CSV progressivamente
            if not os.path.exists(metrics_file_path):
                df_o.to_csv(metrics_file_path, mode='w', index=False)
            else:
                df_new.to_csv(metrics_file_path, mode='a', index=False, header=False)
                
        metrics_file_path = os.path.join(opt.results_dir, 'metrics', f'test_task_model.csv') # percorso del file csv
        if not os.path.exists(metrics_file_path):
            df.to_csv(metrics_file_path, mode='w', index=False)
        else:
            df_new_tm.to_csv(metrics_file_path, mode='a', index=False, header=False)  


if __name__ == '__main__':
    opt = TrainOptions().parse()
    task_model = create_model(opt)  # creo il task model
    dataset = create_dataset(opt)  # ci da il dataloader
    return_layers = opt.return_layers
    compute_tnet_dim(opt)

    if opt.rec_model == 'AENet':
        AE = AENet(opt)
        weights = opt.__dict__.get('weights', [1] * len(AE.AENet))
        for i in range(len(opt.return_layers)):
            name = opt.return_layers[i]
            elem = opt.tnet_dim[i]

            if opt.task == 'denoising':
                load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints2/train_AE/epoch99/AE_{name}_99.pt'

            elif opt.task == 'brats':
                load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints3/train_all_AE/epoch99/AE_{name}_99.pt'

            # cesm
            #load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints4/train_all_AE_cesm/epoch99/AE_{name}_99.pt'
            
            # task IXI
            # load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints3/train_all_AE_IXI/epoch99/AE_{name}_99.pt'

            state_dict = torch.load(load_path_weights, map_location=str(
                AE.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi

            AE.AENet[i].load_state_dict(state_dict)  # così carico il dizionario nel modello
            AE.set_requires_grad(AE.AENet[i], False)

        rec_models = AE.AENet

    if opt.rec_model == 'cycle_gan_paired':
        rec_model = find_model_using_name(opt.rec_model)

        # cycle_gan_paired_0 = rec_model(opt,0)
        rec_model_input = rec_model(opt, 0)
        opt.rec_input_nc = 64
        opt.rec_output_nc = 64
        rec_model_first_conv = rec_model(opt, 1)
        # cycle_gan_paired_1 = rec_model(opt,1)
        opt.rec_input_nc = 256
        opt.rec_output_nc = 256
        rec_model_resnet_block = rec_model(opt, 2)
        # cycle_gan_paired_2 = rec_model(opt,2)
        rec_model_output = rec_model(opt, 3)
        # cycle_gan_paired_3 = rec_model(opt,3)

        load_path_weights_input_G_A = '/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints/prova_train_cycle_gan/epoch99/cycle_gan_rec_input_G_A_99.pt'
        load_path_weights_first_conv_G_A = '/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints/prova_train_cycle_gan_1/epoch99/cycle_gan_rec_first_conv_G_A_99.pt'
        load_path_weights_resnet_block_G_A = '/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints/prova_train_cycle_gan_2/epoch99/cycle_gan_rec_resnet_block_3_G_A_99.pt'
        load_path_weights_output_G_A = '/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints/prova_train_cycle_gan/epoch99/cycle_gan_rec_final_output_G_A_99.pt'

        # TODO: capire se vogliamo vedere la ricpstriuziome dopo il passaggio in Gb
        state_dict_input_G_A = torch.load(load_path_weights_input_G_A, map_location=str(
            rec_model_input.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi.
        state_dict_first_conv_G_A = torch.load(load_path_weights_first_conv_G_A,
                                               map_location=str(rec_model_first_conv.device))
        state_dict_resnet_block_G_A = torch.load(load_path_weights_resnet_block_G_A,
                                                 map_location=str(rec_model_resnet_block.device))
        state_dict_output_G_A = torch.load(load_path_weights_output_G_A, map_location=str(
            rec_model_output.device))  # dizionarioo che ha per chiave il nome del layer

        if isinstance(rec_model_input.netG_A, torch.nn.DataParallel):
            rec_model_input.netG_A.load_state_dict(state_dict_input_G_A)
        else:
            cycle_gan_paired_0.netG_A.module.load_state_dict(state_dict_input_G_A)

        if isinstance(rec_model_first_conv.netG_A, torch.nn.DataParallel):
            rec_model_first_conv.netG_A.load_state_dict(state_dict_first_conv_G_A)
        else:
            rec_model_first_conv.netG_A.module.load_state_dict(state_dict_first_conv_G_A)

        if isinstance(rec_model_resnet_block.netG_A, torch.nn.DataParallel):
            rec_model_resnet_block.netG_A.load_state_dict(state_dict_resnet_block_G_A)
        else:
            rec_model_resnet_block.netG_A.module.load_state_dict(state_dict_resnet_block_G_A)

        if isinstance(rec_model_output.netG_A, torch.nn.DataParallel):
            rec_model_output.netG_A.load_state_dict(state_dict_output_G_A)
        else:
            rec_model_output.netG_A.module.load_state_dict(state_dict_output_G_A)

        rec_model_input.set_requires_grad(
            [rec_model_input.netG_A, rec_model_input.netG_B, rec_model_input.netD_A, rec_model_input.netD_B], False)
        rec_model_first_conv.set_requires_grad(
            [rec_model_first_conv.netG_A, rec_model_first_conv.netG_B, rec_model_first_conv.netD_A,
             rec_model_first_conv.netD_B], False)
        rec_model_resnet_block.set_requires_grad(
            [rec_model_resnet_block.netG_A, rec_model_resnet_block.netG_B, rec_model_resnet_block.netD_A,
             rec_model_resnet_block.netD_B], False)
        rec_model_output.set_requires_grad(
            [rec_model_output.netG_A, rec_model_output.netG_B, rec_model_output.netD_A, rec_model_output.netD_B],
            False)

        rec_models = [rec_model_input, rec_model_first_conv, rec_model_resnet_block, rec_model_output]
        AE = None

    if opt.rec_model == 'pix2pix_rec':
        rec_models = []
        rec_model = find_model_using_name(opt.rec_model)
        for i in range(len(opt.return_layers)):
            name = opt.return_layers[i]
            elem = opt.tnet_dim[i]
            opt.rec_input_nc = elem
            opt.rec_output_nc = elem
            if elem == 1:
                rec_input = False
            else:
                rec_input = True

            rm = rec_model(opt, rec_input)

            load_path_weights = f'/mimer/NOBACKUP/groups/naiss2023-6-336/iele/tesi_Carolina/checkpoints2/train_pix2pix/epoch99/pix2pixrec_{name}_G_99.pt'
            state_dict = torch.load(load_path_weights, map_location=str(
                rm.device))  # dizionario che ha per chiave il nome del layer e per valore i pesi
            if isinstance(rm.netG, torch.nn.DataParallel):
                rm.netG.load_state_dict(state_dict)
            else:
                rm.netG.module.load_state_dict(state_dict)
            rm.set_requires_grad([rm.netG, rm.netD], False)

            rec_models.append(rm)
        AE = None

    adaptors = ANet(opt)

    # carico i pesi del task network
    if opt.task == 'denoising':
        load_path = '/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/pytorch-CycleGAN-and-pix2pix/checkpoints/cycle_gan_paired_modificata_1/100_net_G_A.pth'

    elif opt.task == 'brats':
        load_path = '/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/pytorch-CycleGAN-and-pix2pix/checkpoints/cyclegan_paired_terzo_1/100_net_G_A.pth'

    # cesm
    #load_path = '/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/pytorch-CycleGAN-and-pix2pix/checkpoints/cycle_gan_paired_1_exp2/100_net_G_A.pth'
    
    # task IXI
    # load_path = '/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/pytorch-CycleGAN-and-pix2pix/checkpoints/cyclegan_IXI/100_net_G_A.pth'
    # load_path = f'/mimer/NOBACKUP/groups/snic2022-5-277/cadornato/pytorch-CycleGAN-and-pix2pix/checkpoints/cyclegan_IXI/latest_net_G_A.pth'


    state_dict = torch.load(load_path,
                            map_location=str(task_model.device))  # dizionarioo che ha per chiave il nome del layer
    # e per valore i pesi

    if isinstance(task_model.netG_A, torch.nn.DataParallel):  # se parallelizzo il modello (uso più di 1 gpu)
        task_model.netG_A.module.load_state_dict(
            state_dict)  # parallelizzo ogni rete sulla gpu ( T model ha 4 reti)
    else:
        task_model.netG_A.load_state_dict(state_dict)

    # Carico i pesi del blocco di ricostruzione
    opt_ANet(adaptors, opt, task_model, rec_models, stable=False, model_type=opt.rec_model,
                         AE=AE, return_layers=return_layers)  # adaptor
# NB: l'attributo AENet ci serve per usare la funzione AELoss, e per concatenare le feature. Se non usiamo AENet, AENet=None