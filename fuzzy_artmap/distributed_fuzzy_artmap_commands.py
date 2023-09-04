class DistributedFuzzyArtmapCommands():
    def __init__(self) -> None:
        self.init_header = "i".encode("utf-8")
        self.get_worker_address_header = "g".encode("utf-8")
        self.predict_header = "p".encode("utf-8")
        self.fit_header = "f".encode("utf-8")
        self.payload_seperator = "|".encode("utf-8")
        self.prediction_response_header = "r".encode("utf-8")
        self.end_mark = "|||".encode("utf-8")
        self.error_header = "e".encode("utf-8")
        self.protocol_overhead = 8
