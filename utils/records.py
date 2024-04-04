def record_model_metrics(writer, phase, epoch: int, epoch_loss, all_metrics: dict):
    writer.add_scalars('Loss', {
        phase: epoch_loss,
    }, epoch)
    for key in all_metrics.keys():
        writer.add_scalars(key, {
            phase: all_metrics[key],
        }, epoch)
    # writer.add_scalars('acc', {
    #     phase: all_metrics['acc'],
    # }, epoch)
    # writer.add_scalars('mcc', {
    #     phase: all_metrics['mcc'],
    # }, epoch)
    # writer.add_scalars('precision', {
    #     phase: all_metrics['precision'],
    # }, epoch)
    # writer.add_scalars('recall', {
    #     phase: all_metrics['recall'],
    # })
    # writer.add_scalars('f1', {
    #     phase: all_metrics['f1'],
    # })
    # writer.add_scalars('tpr', {
    #     phase: all_metrics['tpr'],
    # })
    # writer.add_scalars('fpr', {
    #     phase: all_metrics['fpr'],
    # })
    # writer.add_scalars('ks', {
    #     phase: all_metrics['ks'],
    # }, epoch)
    # writer.add_scalars('sp', {
    #     phase: all_metrics['sp'],
    # }, epoch)
    writer.flush()
