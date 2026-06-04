from concurrent.futures import ThreadPoolExecutor

from django.db import close_old_connections

from studies.services.analysis_runner import process_study_analysis_run


_analysis_executor = ThreadPoolExecutor(max_workers=1)


def queue_study_analysis_run(run_id: int):
	"""
	Queues a machine analysis run in a background thread.
	
	Note for future self:
	This is good enough for local/dev/prototype use.
	For production, I'll replace this with Celery/RQ/Django-Q.
	"""
	return _analysis_executor.submit(_run_analysis_safely, run_id)


def _run_analysis_safely(run_id: int):
	close_old_connections()

	try:
		process_study_analysis_run(run_id)
	finally:
		close_old_connections()