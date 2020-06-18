import click
import os
import pickle
import datetime
from sklearn.metrics import f1_score, accuracy_score
from ginipls.data.data_utils import load_data, save_y_in_file, save_ytrue_and_ypred_in_file, load_ytrue_ypred_file
from ginipls.models.ginipls import PLS, PLS_VARIANT
from ginipls.models.hyperparameters import select_pls_hyperparameters_with_cross_val
from ginipls.config import GLOBAL_LOGGER as logger

DEFAULT_PREDICTION_DIR = 'data/predictions'
DEFAULT_MODELS_DIR = 'models'
#CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

def cast_click_list(value):
    try:
        print('value = ', value, "type(value)", type(value))
        e = ""
        real_list = list()
        for c in value:
            if c == ' ' or c==',' or c=='[' or c==']':
                if e != "":
                    real_list.append(e)
                e = ""
                continue
            e+=c
        if e != "":
            real_list.append(e)
        #print("real_list", real_list)
        return real_list
    except Exception:
        raise click.BadParameter(value)

def init_and_train_pls(X_train, y_train, pls_type, hyerparameters_selection_nfolds, nu_range, n_components_range, only_the_first_fold):
  """"""
  best_nu, best_n_comp = select_pls_hyperparameters_with_cross_val(pls_type, X_train, y_train, nu_range, n_components_range, hyerparameters_selection_nfolds, only_the_first_fold=only_the_first_fold)
  logger.info("selected hyperparameters : nu=%.3f, n_comp=%d" % (best_nu, best_n_comp))
  gpls = PLS(pls_type=pls_type, nu=best_nu, n_components=best_n_comp)
  gpls.fit(X_train, y_train)
  train_score = gpls.score(X_train, y_train)
  logger.info("train f1_score = %.3f" % (train_score))
  return gpls

def train_on_vectors(trainfilename, classifierfilename, label_col, index_col, col_sep, pls_type, nu_range, n_components_range, hyperparams_nfolds, crossval_hyperparam):
    """python -m ginipls train on-vectors --label_col=category --index_col=@id --crossval_hyperparam  data\processed\doris0_CHI2_ATF-train.tsv"""
    if not isinstance(n_components_range[0], int):
        n_components_range = [int(x) for x in cast_click_list(n_components_range)]
    click.echo('This is the on-vectors subcommand of the train command : nu_range=%s, n_components_range=%s' % (nu_range, n_components_range))
    X_train, y_train, headers, ids = load_data(data=trainfilename, output_col=label_col, index_col=index_col, col_sep=col_sep)
    clf = init_and_train_pls(X_train, y_train, pls_type, hyperparams_nfolds, nu_range, n_components_range, only_the_first_fold=not crossval_hyperparam)
    if classifierfilename is None:
        # build a name for the model_file
        input_basename = os.path.basename(trainfilename).split('.')[0]
        if not os.path.isdir(DEFAULT_MODELS_DIR):
            os.mkdir(DEFAULT_MODELS_DIR)
        classifierfilename = os.path.join(DEFAULT_MODELS_DIR,"_".join([datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),input_basename, str(pls_type.name)+"PLS.model"]))
    pickle.dump(clf, open(classifierfilename, 'wb'))
    logger.info("The trained classifier is saved at %s" % (classifierfilename))

def apply_on_vectors(vectorsfilename, classifierfilename, outputfilename, label_col, index_col, col_sep):
    """python -m ginipls apply on-vectors --label_col=category --index_col=@id data\processed\doris0_CHI2_ATF-test.tsv models\2020-06-11_14-58-50_doris0_CHI2_ATF-train_GINIPLS.model"""
    click.echo('This is the on-vectors subcommand of the apply command')
    X_test, y_test, h, ids_test = load_data(data=vectorsfilename, output_col=label_col, index_col=index_col, col_sep=col_sep)
    clf = pickle.load(open(classifierfilename, 'rb'))
    #logger.debug(clf.get_params())
    logger.info("Trained %s PLS classifier loaded from %s." % (clf.get_params()["pls_type"].name, classifierfilename))
    y_pred = clf.predict(X_test)
    if outputfilename is None:
        # build a name for the model_file
        vector_fbasename = os.path.basename(vectorsfilename).split('.')[0]
        model_fbasename = os.path.basename(classifierfilename).split('.')[0]
        if not os.path.isdir(DEFAULT_PREDICTION_DIR):
            os.mkdir(DEFAULT_PREDICTION_DIR)
        outputfilename = os.path.join(DEFAULT_PREDICTION_DIR,"_".join([model_fbasename, 'applied-on', vector_fbasename, "out.tsv"]))
    if label_col: # save y_pred and y_true
        save_ytrue_and_ypred_in_file(row_index=ids_test, y_trues=y_test, y_preds=y_pred, y_file_name=outputfilename, col_sep=col_sep)
    else:
        save_y_in_file(row_index=ids_test, y=y_pred, y_file_name=outputfilename, col_sep=col_sep)
    logger.info("Predicted labels are saved at %s" % (outputfilename))


@click.group(help = "train models on labeled data")
def train():
    pass

@train.command("on-vectors", help="train on a csv file with vectors as rows and attributes as columns")
@click.argument('trainfilename', type=click.Path(exists=True))
@click.argument('classifierfilename', type=click.Path(), required=False)
@click.option('--label_col', type=str, default='@label', help='labels column name', show_default=True)
@click.option('--index_col', type=str, default=None, help='texts ids column name[optional]', show_default=True, required=False)
@click.option('--col_sep', type=str, default="\t", help='column delimiter', show_default=True)
@click.option('--pls_type', type=click.Choice(PLS_VARIANT.__members__), callback=lambda c, p, v: getattr(PLS_VARIANT, v) if v else None, default=PLS_VARIANT.STANDARD, help='variant of PLS (%s)' % (' '.join('PLS_VARIANT.' + c.name for c in PLS_VARIANT)), show_default=True)
@click.option('--nu_range', type=list, default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], help='range of the hyperparameter nu of the gini_pls method', show_default=True)
@click.option('--n_components_range', type=list, default=" ".join([str(n) for n in range(1,5)]), help='range of the hyperparameter n_components', show_default=True)
@click.option('--hyperparams_nfolds', type=int, default=3, help='nb of folds for the selection of hyperparameters by cross-validation', show_default=True)
@click.option('--crossval_hyperparam/--no-crossval_hyperparam', default=False, help='run on all the fold or only on the first fold for the selection of hyperparameters', show_default=True)
def on_vectors(trainfilename, classifierfilename, label_col, index_col, col_sep, pls_type, nu_range, n_components_range, hyperparams_nfolds, crossval_hyperparam):
    """python -m ginipls train on-vectors --label_col=category --index_col=@id --crossval_hyperparam  data\processed\doris0_CHI2_ATF-train.tsv"""
    train_on_vectors(trainfilename, classifierfilename, label_col, index_col, col_sep, pls_type, nu_range, n_components_range, hyperparams_nfolds, crossval_hyperparam)

@train.command("on-texts", help="vectorize training texts then train on their vectors")
def train_on_texts():
    click.echo('This is the on-texts subcommand of the train command [UNAVAILABLE]')

#cli = click.CommandCollection(sources=[train])

@click.group(help = "apply a trained model on new data")
def apply():
    pass
    
@apply.command('on-vectors', help="apply a trained pls on vectors and save outputs in outputfilename")
@click.argument('vectorsfilename', type=click.Path(exists=True))
@click.argument('classifierfilename', type=click.Path(exists=True))
@click.argument('outputfilename', type=click.Path(), required=False)
@click.option('--label_col', type=str, default=None, help='labels column name [optional]', show_default=True, required=False)
@click.option('--index_col', type=str, default=None, help='texts ids column name[optional]', show_default=True, required=False)
@click.option('--col_sep', type=str, default="\t", help='column delimiter', show_default=True)
def on_vectors(vectorsfilename, classifierfilename, outputfilename, label_col, index_col, col_sep):
    """python -m ginipls apply on-vectors --label_col=category --index_col=@id data\processed\doris0_CHI2_ATF-test.tsv models\2020-06-11_14-58-50_doris0_CHI2_ATF-train_GINIPLS.model"""
    apply_on_vectors(vectorsfilename, classifierfilename, outputfilename, label_col, index_col, col_sep)

@apply.command('on-text', help="vectorize training texts then apply a trained pls on their vectors")
def on_texts():
    click.echo('This is the on-texts subcommand of the train command [UNAVAILABLE]')

@click.group(help = "apply a trained model on new data and estimate the prediction score")
def test():
    pass
    

@test.command('on-vectors', help="apply a trained model on vectors")
def test_on_vectors(trainfilename, classifierfilename, label_col, index_col, col_sep):
    # test_acc_score = accuracy_score(y_test, y_pred)
    # logger.info("test accuracy_score = %.3f" % (test_acc_score))
    # test_f1_score = f1_score(y_test, y_pred, labels=[0,1], average='macro')
    # logger.info("test f1_score = %.3f" % (test_f1_score))
    pass
    
@click.group(help = "evaluate a prediction")
def evaluate():
    pass

@evaluate.command('f1', help="estimate the F1-score on two Y vectors stored in predfilename (ytrue, ypred)")
@click.argument('predfilename', type=click.Path(exists=True))
@click.option('--index_col', type=str, default=None, help='texts ids column name [optional]', show_default=True, required=False)
@click.option('--ytrue_col', type=str, default=None, help='column of expected output', show_default=True, required=False)
@click.option('--ypred_col', type=str, default=None, help='column of predicted output', show_default=True, required=False)
@click.option('--col_sep', type=str, default="\t", help='column delimiter', show_default=True)
def f1_score_on_prediction_file(predfilename, index_col, ytrue_col, ypred_col, col_sep):
    _, y_true, y_pred = load_ytrue_ypred_file(y_file_name=predfilename, indexCol=index_col, yTrueCol=ytrue_col, yPredCol=ypred_col, col_sep=col_sep)
    test_f1_score = f1_score(y_true, y_pred, labels=[0,1], average='macro')
    if logger.disabled:
        print(test_f1_score)
    logger.info("test f1_score = %.3f" % (test_f1_score))

@evaluate.command('accuracy', help="apply a trained pls on vectors")
@click.argument('predfilename', type=click.Path(exists=True))
@click.option('--index_col', type=str, default=None, help='texts ids column name [optional]', show_default=True, required=False)
@click.option('--ytrue_col', type=str, default=None, help='column of expected output', show_default=True, required=False)
@click.option('--ypred_col', type=str, default=None, help='column of predicted output', show_default=True, required=False)
@click.option('--col_sep', type=str, default="\t", help='column delimiter', show_default=True)
def accuracy_score_on_prediction_file(predfilename, index_col, ytrue_col, ypred_col, col_sep):
    _, y_true, y_pred = load_ytrue_ypred_file(y_file_name=predfilename, indexCol=index_col, yTrueCol=ytrue_col, yPredCol=ypred_col, col_sep=col_sep)
    test_acc_score = accuracy_score(y_true, y_pred)
    if logger.disabled:
        print(test_acc_score)
    logger.info("test accuracy_score = %.3f" % (test_acc_score))


@click.group(help = "This is the command line interface of the Gini-PLS classification project")
@click.option('--logging/--no-logging', default=True, help='column delimiter', show_default=True)
def cli(logging):
    logger.disabled = (not logging)

cli.add_command(apply)
cli.add_command(evaluate)
cli.add_command(test)
cli.add_command(train)

if __name__ == '__main__':
# python -m ginipls --no-logging evaluate f1 data\predictions\2020-06-11_14-58-50_doris0_CHI2_ATF-train_GINIPLS_applied-on_doris0_CHI2_ATF-test_out.tsv
    cli()
