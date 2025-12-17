"""
Spring API 엔드포인트 관리 모듈

이 모듈은 Spring API 서버의 모든 엔드포인트 경로를 중앙 집중식으로 관리합니다.
엔드포인트 변경 시 이 파일만 수정하면 됩니다.

클린 코드 원칙:
- 단일 책임: API 엔드포인트 경로만 관리
- 개방-폐쇄: 새로운 엔드포인트 추가가 용이
- 중앙 집중식 관리: 변경 지점 최소화
"""


class SpringAPIEndpoints:
    """Spring API 엔드포인트 경로 관리 클래스"""
    
    # API 기본 경로
    API_BASE = "/api"
    
    # Job 관련 엔드포인트
    JOBS_BASE = f"{API_BASE}/jobs"
    
    @staticmethod
    def job_callback(job_id: str) -> str:
        """
        작업 결과 콜백 엔드포인트
        
        Args:
            job_id: 작업 ID
            
        Returns:
            콜백 엔드포인트 경로 (예: /api/jobs/{job_id}/callback)
        """
        return f"{SpringAPIEndpoints.JOBS_BASE}/{job_id}/callback"
    
    @staticmethod
    def job_status(job_id: str) -> str:
        """
        작업 상태 조회 엔드포인트
        
        Args:
            job_id: 작업 ID
            
        Returns:
            상태 조회 엔드포인트 경로 (예: /api/jobs/{job_id}/status)
        """
        return f"{SpringAPIEndpoints.JOBS_BASE}/{job_id}/status"
    
    @staticmethod
    def health_check() -> str:
        """
        헬스 체크 엔드포인트
        
        Returns:
            헬스 체크 엔드포인트 경로 (예: /api/health)
        """
        return f"{SpringAPIEndpoints.API_BASE}/health"

