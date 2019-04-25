from PyQt5.QtWidgets import QMainWindow, QFileDialog
from window import Ui_MainWindow
from rec_sys.semantic import semantic_distance
from rec_sys.colaboratif import colaboratif_distance
from rec_sys.hybrid import WHybrid, MixHybrid
from rec_sys.read_csv import load_data
from rec_sys.metrics import save_dist_matrix, load_dist_matrix
#from rec_sys.clustering.dbscan import dbscan
from pyclustering.cluster import dbscan

class ApplicationWindow(QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.udm = None

        self.ui.base_data_btn.clicked.connect(self.on_base_btn_clicked)
        self.ui.user_data_btn.clicked.connect(self.on_user_btn_clicked)
        self.ui.item_data_btn.clicked.connect(self.on_item_btn_clicked)
        self.ui.load_data_btn.clicked.connect(self.on_load_data)
        self.ui.process_btn.clicked.connect(self.on_process)
        self.ui.save_btn.clicked.connect(self.on_save_matrix)
        self.ui.load_matrix_btn.clicked.connect(self.on_load_matrix)
        self.ui.run_dbscan_btn.clicked.connect(self.on_run_dbscan)

        self.ui.filtering_methods_cb.addItems(
            [
                "Colabortive filtering",
                "Semantic filtering",
                "Weighted Hybrid filtering",
                "Semantic Colabrative Hybrid filtering"
            ]
        )

        
    def on_base_btn_clicked(self):
        self.ui.base_data_path.setText(
            QFileDialog.getOpenFileName(self, "Base Dataset File", "/home/imad", "base (*.base)")[0]
        )
        print('[input file]', self.ui.base_data_path.text())
    
    def on_item_btn_clicked(self):
        self.ui.item_data_path.setText(
            QFileDialog.getOpenFileName(self, "item File", "/home/imad", "item (*.item)")[0]
        )
        print('[input file]', self.ui.item_data_path.text())

    def on_user_btn_clicked(self):
        self.ui.user_data_path.setText(
            QFileDialog.getOpenFileName(self, "Base Dataset File", "/home/imad", "user (*.user)")[0]
        )
        print('[input file]', self.ui.user_data_path.text())

    def on_load_data(self):
        print('[load data] started')
        self.ui.status_label.setText('Loading data...')
        self.usage_matrix, self.movie_matrix = load_data(
            self.ui.base_data_path.text(),
            self.ui.user_data_path.text(),
            self.ui.item_data_path.text()
        )
        print('[load data] finished')
        self.ui.status_label.setText('Done')

    def on_process(self):
        self.ui.status_label.setText('Processing...')
        print('[processing]', self.ui.filtering_methods_cb.currentText())
        self.udm = self.filter()
        self.ui.status_label.setText('Done')
        print('[Done]', self.ui.filtering_methods_cb.currentText())
        print(self.udm)
    
    def filter(self):
        if self.ui.filtering_methods_cb.currentIndex() == 0:
            return colaboratif_distance(self.usage_matrix)
        elif self.ui.filtering_methods_cb.currentIndex() == 1:
            return semantic_distance(self.usage_matrix, self.movie_matrix)
        elif self.ui.filtering_methods_cb.currentIndex() == 2:
            return None
        elif self.ui.filtering_methods_cb.currentIndex() == 3:
            return None
    
    def on_save_matrix(self):
        if self.udm is None:
            self.ui.status_label.setText("<font color='red'>Matrix isn't ready to save</font>")
            return None
        filename = QFileDialog.getSaveFileName(self, 'Save Matrix')
        print(filename)
        if filename is not None: save_dist_matrix(filename[0], self.udm)

    def on_load_matrix(self):
        self.ui.matrix_file_path.setText(
            QFileDialog.getOpenFileName(self, "CSV distance matrix", "/home/imad", "csv (*.csv)")[0]
        )
        print('[input file]', self.ui.matrix_file_path.text())


    def on_run_dbscan(self):
        if self.ui.matrix_file_path.text() == '':
            self.ui.status_label.setText("<font color='red'>No data specified</font>")
            return None
        self.ui.status_label.setText("Loading data...")
        self.udm = load_dist_matrix(self.ui.matrix_file_path.text())
        self.ui.status_label.setText("Running dbscan for :{}...".format(self.ui.matrix_file_path.text()))
        dbs = dbscan.dbscan(
            self.udm,
            float(self.ui.eps_edit.text()),
            int(self.ui.min_points_edit.text()),
            data_type="distance_matrix"
        )
        dbs.process()
        
        self.ui.status_label.setText("Done.")
        print(dbs.get_clusters)