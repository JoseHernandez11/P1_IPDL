from accelerate import Accelerator
# ... (todas tus importaciones anteriores)

# Inicializamos Accelerator
accelerator = Accelerator(cpu=True)
device = accelerator.device
print(f"Using device: {device}")

# Adaptamos `get_dataloader` para mover tensores a dispositivo
def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'spa', True)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(target_ids))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

# Adaptamos la función de entrenamiento por epoch con accelerator
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        accelerator.backward(loss)

        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Adaptamos la función principal `train` para usar `accelerator.prepare`
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100, save_directory=None):
    start = time.time()
    plot_losses = []
    epoch_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Preparamos todo con accelerate
    encoder, decoder, encoder_optimizer, decoder_optimizer, train_dataloader, criterion = accelerator.prepare(
        encoder, decoder, encoder_optimizer, decoder_optimizer, train_dataloader, criterion
    )

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        epoch_losses.append(loss)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # Guardado condicional (igual que antes)
        save_flag = False
        if len(epoch_losses) > 2:
            if epoch_losses[-1] < min(epoch_losses[:-1]):
                print(f"La época {epoch} ha mejorado la anterior. Prev loss: {min(epoch_losses[:-1])}, current loss: {epoch_losses[-1]}")
                save_flag = True

        if save_directory and (epoch == 1 or epoch % 20 == 0 or save_flag):
            os.makedirs(save_directory, exist_ok=True)
            if epoch == 1 or save_flag:
                encoder_path = os.path.join(save_directory, 'encoder_best_model.pt')
                decoder_path = os.path.join(save_directory, 'decoder_best_model.pt')
            elif epoch % 20 == 0:
                encoder_path = os.path.join(save_directory, f'encoder_{epoch}_model.pt')
                decoder_path = os.path.join(save_directory, f'decoder_{epoch}_model.pt')
            torch.save(accelerator.unwrap_model(encoder).state_dict(), encoder_path)
            torch.save(accelerator.unwrap_model(decoder).state_dict(), decoder_path)

    return epoch_losses
