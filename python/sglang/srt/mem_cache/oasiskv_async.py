"""Independent graph launches for a layer's cross-step prefetch pipeline.

External event nodes connect three separately captured graphs. The foreground
does not fork capture onto a side stream, so it has no graph-tail join. The host
launches background graphs immediately after submitting the foreground graph;
GPU event waits release each layer as soon as its producers finish.
"""

import torch


class LayerPrefetchGraphs:
    def __init__(self, select_stream, transfer_stream):
        self.select_stream = select_stream
        self.transfer_stream = transfer_stream
        self.inputs_ready = torch.cuda.Event(external=True)
        self.prediction_ready = torch.cuda.Event(external=True)
        self.read_done = torch.cuda.Event(external=True)
        self.selection_done = torch.cuda.Event(external=True)
        self.done = torch.cuda.Event(external=True)
        self.graphs = {}
        self.functions = {}
        self.pending = False
        for event in self.events:
            event.record()

    @property
    def events(self):
        return (
            self.inputs_ready,
            self.prediction_ready,
            self.read_done,
            self.selection_done,
            self.done,
        )

    def consume(self):
        # These waits become external nodes in the foreground graph. They
        # consume the preceding launch, before this layer overwrites scratch.
        self.inputs_ready.wait()
        self.done.wait()

    def capture(self, shape, select, transfer):
        if shape in self.graphs:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "OasisKV side graphs must be warmed before model capture"
            )
        # Only startup/recapture reaches here, with inactive request rows.
        # Compile kernels before capturing the independent side graphs.
        # Tables and scratch were allocated/initialized on the caller stream.
        self.synchronize()
        self.select_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.select_stream):
            select()
        self.select_stream.synchronize()
        with torch.cuda.stream(self.transfer_stream):
            transfer()
        self.transfer_stream.synchronize()
        selection, copying = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(selection, stream=self.select_stream):
            self.prediction_ready.wait()
            select()
            self.selection_done.record()
        with torch.cuda.graph(copying, stream=self.transfer_stream):
            self.selection_done.wait()
            self.read_done.wait()
            transfer()
            self.done.record()
        self.graphs[shape] = (selection, copying)
        self.functions[shape] = (select, transfer)

    def replay(self, shape):
        selection, copying = self.graphs[shape]
        with torch.cuda.stream(self.select_stream):
            selection.replay()
        with torch.cuda.stream(self.transfer_stream):
            copying.replay()
        self.pending = True

    def synchronize(self):
        if self.pending:
            self.done.synchronize()
            self.pending = False


class PrefetchGraphBatch:
    """Two launches for all layers, retaining every layer's external event nodes.

    The select and transfer streams already order layers serially. Capturing
    those same sequences together removes per-layer host launches, without a
    foreground graph-tail join or a wait for all transfers before the next step.
    """

    def __init__(self):
        self.graphs = {}

    def capture(self, shape, pipes):
        if shape in self.graphs or not pipes:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "OasisKV prefetch batch must be warmed before model capture"
            )
        select_stream, transfer_stream = (
            pipes[0].select_stream,
            pipes[0].transfer_stream,
        )
        for pipe in pipes:
            if (
                pipe.select_stream != select_stream
                or pipe.transfer_stream != transfer_stream
            ):
                raise ValueError("OasisKV prefetch batch requires shared stage streams")
            pipe.synchronize()
        select_stream.wait_stream(torch.cuda.current_stream())
        selection, copying = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(selection, stream=select_stream):
            for pipe in pipes:
                pipe.prediction_ready.wait()
                pipe.functions[shape][0]()
                pipe.selection_done.record()
        with torch.cuda.graph(copying, stream=transfer_stream):
            for pipe in pipes:
                pipe.selection_done.wait()
                pipe.read_done.wait()
                pipe.functions[shape][1]()
                pipe.done.record()
        self.graphs[shape] = (selection, copying, tuple(pipes))

    def replay(self, shape):
        selection, copying, pipes = self.graphs[shape]
        with torch.cuda.stream(pipes[0].select_stream):
            selection.replay()
        with torch.cuda.stream(pipes[0].transfer_stream):
            copying.replay()
        for pipe in pipes:
            pipe.pending = True
