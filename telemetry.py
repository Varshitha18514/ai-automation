
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)

resource = Resource.create({
    "service.name": "ai-automation-rag"
})

provider = TracerProvider(resource=resource)

exporter = ConsoleSpanExporter()

processor = SimpleSpanProcessor(exporter)

provider.add_span_processor(processor)

trace.set_tracer_provider(provider)

tracer = trace.get_tracer("rag_pipeline")
