"""
Resources Module - All advanced features consolidated
Includes: Structured Extraction, Audit Trail, Query Cache, Knowledge Base Setup
"""

import re
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ============================================================================
# STRUCTURED DATA EXTRACTION
# ============================================================================

class ProcurementExtractor:
    """Extract structured procurement data from document chunks"""
    
    # Regex patterns for RA 9184 fields
    PATTERNS = {
        'abc': [
            r'ABC[:\s]*(?:PHP|Php|P|₱)?\s*([\d,]+\.?\d*)',
            r'Approved Budget.*?(?:PHP|Php|P|₱)?\s*([\d,]+\.?\d*)',
            r'Budget.*?Contract.*?(?:PHP|Php|P|₱)?\s*([\d,]+\.?\d*)',
        ],
        'pr_number': [
            r'PR\s*(?:Number|No\.?|#)?[:\s]*(\d+[-\d]*)',
            r'Purchase\s+Request.*?(?:Number|No\.?|#)?[:\s]*(\d+[-\d]*)',
        ],
        'delivery_period': [
            r'(\d+)\s*(?:calendar|working)?\s*days?',
            r'Delivery.*?Period[:\s]*(\d+)\s*(?:calendar|working)?\s*days?',
            r'within\s*(\d+)\s*(?:calendar|working)?\s*days?',
        ],
        'pre_bid_conference': [
            r'Pre[-\s]?Bid.*?Conference[:\s]*(.*?)(?:\n|$)',
            r'Pre[-\s]?Procurement.*?Conference[:\s]*(.*?)(?:\n|$)',
        ],
        'bid_opening': [
            r'Bid\s+Opening[:\s]*(.*?)(?:\n|$)',
            r'Opening\s+of\s+Bids?[:\s]*(.*?)(?:\n|$)',
        ],
        'closing_date': [
            r'Closing\s+Date[:\s]*(.*?)(?:\n|$)',
            r'Deadline[:\s]*(.*?)(?:\n|$)',
            r'Submission.*?Deadline[:\s]*(.*?)(?:\n|$)',
        ],
        'contract_amount': [
            r'Contract\s+(?:Amount|Price|Value)[:\s]*(?:PHP|Php|P|₱)?\s*([\d,]+\.?\d*)',
        ],
        'supplier': [
            r'(?:Supplier|Contractor|Bidder)[:\s]*([A-Z][A-Za-z\s&.,]+?)(?:\n|,|$)',
        ],
        'item_description': [
            r'Item\s+Description[:\s]*(.*?)(?:\n|Item|$)',
            r'Description[:\s]*(.*?)(?:\n|Quantity|Unit|$)',
        ],
    }
    
    @staticmethod
    def extract_amount(text: str) -> Optional[float]:
        """Extract monetary amount from text"""
        for pattern in ProcurementExtractor.PATTERNS['abc']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except:
                    continue
        return None
    
    @staticmethod
    def extract_pr_number(text: str) -> Optional[str]:
        """Extract Purchase Request number"""
        for pattern in ProcurementExtractor.PATTERNS['pr_number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    @staticmethod
    def extract_delivery_period(text: str) -> Optional[str]:
        """Extract delivery period in days"""
        for pattern in ProcurementExtractor.PATTERNS['delivery_period']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} Calendar Days"
        return None
    
    @staticmethod
    def extract_date_time(text: str) -> Optional[str]:
        """Extract date and time from text"""
        date_patterns = [
            r'(\w+ \d{1,2}, \d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\w+-\d{4})',
        ]
        
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',
        ]
        
        date_str = None
        time_str = None
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                break
        
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                time_str = match.group(1)
                break
        
        if date_str and time_str:
            return f"{date_str} {time_str}"
        elif date_str:
            return date_str
        return None
    
    @staticmethod
    def extract_compliance_fields(chunks: List[Dict]) -> Dict:
        """Extract all RA 9184 compliance fields from document chunks"""
        combined_text = "\n".join([chunk['content'] for chunk in chunks])
        
        result = {
            'abc': None,
            'pr_number': None,
            'delivery_period': None,
            'pre_bid_conference': None,
            'bid_opening': None,
            'closing_date': None,
            'sources': {}
        }
        
        # Extract ABC
        abc_amount = ProcurementExtractor.extract_amount(combined_text)
        if abc_amount:
            result['abc'] = f"PHP {abc_amount:,.2f}"
            for chunk in chunks:
                if ProcurementExtractor.extract_amount(chunk['content']):
                    result['sources']['abc'] = {
                        'page': chunk.get('page', 0),
                        'source': chunk.get('source', '')
                    }
                    break
        
        # Extract PR Number
        pr_num = ProcurementExtractor.extract_pr_number(combined_text)
        if pr_num:
            result['pr_number'] = pr_num
            for chunk in chunks:
                if ProcurementExtractor.extract_pr_number(chunk['content']):
                    result['sources']['pr_number'] = {
                        'page': chunk.get('page', 0),
                        'source': chunk.get('source', '')
                    }
                    break
        
        # Extract Delivery Period
        delivery = ProcurementExtractor.extract_delivery_period(combined_text)
        if delivery:
            result['delivery_period'] = delivery
            for chunk in chunks:
                if ProcurementExtractor.extract_delivery_period(chunk['content']):
                    result['sources']['delivery_period'] = {
                        'page': chunk.get('page', 0),
                        'source': chunk.get('source', '')
                    }
                    break
        
        # Extract dates
        for field in ['pre_bid_conference', 'bid_opening', 'closing_date']:
            for pattern in ProcurementExtractor.PATTERNS[field]:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    date_info = match.group(1).strip()
                    extracted_date = ProcurementExtractor.extract_date_time(date_info)
                    if extracted_date:
                        result[field] = extracted_date
                        for chunk in chunks:
                            if date_info in chunk['content']:
                                result['sources'][field] = {
                                    'page': chunk.get('page', 0),
                                    'source': chunk.get('source', '')
                                }
                                break
                    break
        
        return result
    
    @staticmethod
    def compare_documents(doc1_data: Dict, doc2_data: Dict) -> Dict:
        """Compare procurement data between two documents"""
        comparison = {
            'abc_difference': None,
            'delivery_period_difference': None,
            'timeline_comparison': {},
            'missing_fields': {
                'doc1': [],
                'doc2': []
            }
        }
        
        # Compare ABC amounts
        if doc1_data.get('abc') and doc2_data.get('abc'):
            try:
                amt1 = float(doc1_data['abc'].replace('PHP', '').replace(',', '').strip())
                amt2 = float(doc2_data['abc'].replace('PHP', '').replace(',', '').strip())
                comparison['abc_difference'] = amt1 - amt2
            except:
                pass
        
        # Compare delivery periods
        if doc1_data.get('delivery_period') and doc2_data.get('delivery_period'):
            try:
                days1 = int(re.search(r'(\d+)', doc1_data['delivery_period']).group(1))
                days2 = int(re.search(r'(\d+)', doc2_data['delivery_period']).group(1))
                comparison['delivery_period_difference'] = days1 - days2
            except:
                pass
        
        # Check missing fields
        fields = ['abc', 'pr_number', 'delivery_period', 'pre_bid_conference', 'bid_opening', 'closing_date']
        for field in fields:
            if not doc1_data.get(field):
                comparison['missing_fields']['doc1'].append(field)
            if not doc2_data.get(field):
                comparison['missing_fields']['doc2'].append(field)
        
        return comparison


# ============================================================================
# AUDIT TRAIL SYSTEM
# ============================================================================

class AuditTrail:
    """Track all queries and retrievals for compliance and debugging"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audit_dir = Path('./sessions') / session_id / 'audit'
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.audit_dir / 'query_log.jsonl'
    
    def log_query(
        self,
        query: str,
        response: str,
        chunks: List[Dict],
        debug_info: Dict,
        selected_document: Optional[str] = None
    ) -> None:
        """Log a query with full context"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'query': query,
            'response': response,
            'selected_document': selected_document,
            'retrieval': {
                'total_chunks': len(chunks),
                'document_chunks': sum(1 for c in chunks if c.get('type') == 'document'),
                'kb_chunks': sum(1 for c in chunks if c.get('type') == 'permanent_knowledge'),
                'chunks_detail': [
                    {
                        'content_preview': chunk['content'][:200] + '...' if len(chunk['content']) > 200 else chunk['content'],
                        'source': chunk.get('source', 'N/A'),
                        'page': chunk.get('page', 0),
                        'score': chunk.get('score', 0),
                        'type': chunk.get('type', 'document')
                    }
                    for chunk in chunks
                ]
            },
            'performance': {
                'retrieval_time': debug_info.get('retrieval_time', 0),
                'generation_time': debug_info.get('generation_time', 0),
                'total_time': debug_info.get('total_time', 0),
                'input_tokens': debug_info.get('input_tokens', 0),
                'output_tokens': debug_info.get('output_tokens', 0)
            }
        }
        
        with open(self.audit_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def get_query_history(self, limit: int = 50) -> List[Dict]:
        """Retrieve recent query history"""
        if not self.audit_file.exists():
            return []
        
        entries = []
        with open(self.audit_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except:
                    continue
        
        return list(reversed(entries[-limit:]))
    
    def search_queries(self, keyword: str) -> List[Dict]:
        """Search audit trail for queries containing keyword"""
        if not self.audit_file.exists():
            return []
        
        matches = []
        keyword_lower = keyword.lower()
        
        with open(self.audit_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if keyword_lower in entry['query'].lower() or keyword_lower in entry['response'].lower():
                        matches.append(entry)
                except:
                    continue
        
        return matches
    
    def get_document_usage(self) -> Dict[str, int]:
        """Get statistics on which documents have been queried"""
        if not self.audit_file.exists():
            return {}
        
        usage = {}
        
        with open(self.audit_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    for chunk in entry.get('retrieval', {}).get('chunks_detail', []):
                        source = chunk.get('source', 'Unknown')
                        usage[source] = usage.get(source, 0) + 1
                except:
                    continue
        
        return dict(sorted(usage.items(), key=lambda x: x[1], reverse=True))
    
    def export_audit_report(self) -> str:
        """Generate a human-readable audit report"""
        entries = self.get_query_history(limit=100)
        
        report = f"# Audit Trail Report\n\n"
        report += f"**Session ID:** {self.session_id}\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Total Queries:** {len(entries)}\n\n"
        
        report += "## Document Usage Statistics\n\n"
        usage = self.get_document_usage()
        for doc, count in usage.items():
            report += f"- {doc}: {count} retrievals\n"
        
        report += "\n## Query History (Most Recent)\n\n"
        for i, entry in enumerate(entries[:20], 1):
            report += f"### Query {i}\n"
            report += f"**Time:** {entry['timestamp']}\n"
            report += f"**Question:** {entry['query']}\n"
            report += f"**Chunks Retrieved:** {entry['retrieval']['total_chunks']}\n"
            report += f"**Response Time:** {entry['performance']['total_time']:.2f}s\n\n"
        
        return report


# ============================================================================
# QUERY CACHE SYSTEM
# ============================================================================

class QueryCache:
    """Cache query results with semantic similarity matching"""
    
    def __init__(self, session_id: str, embedding_model, cache_ttl_hours: int = 24):
        self.session_id = session_id
        self.embedding_model = embedding_model
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        self.cache_dir = Path('./sessions') / session_id / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_index_file = self.cache_dir / 'index.json'
        self.cache_index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load cache index from disk"""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'entries': []}
    
    def _save_index(self) -> None:
        """Save cache index to disk"""
        with open(self.cache_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _generate_cache_key(self, query: str, selected_document: Optional[str] = None) -> str:
        """Generate unique cache key for query"""
        cache_input = f"{query}|{selected_document or 'all'}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _is_expired(self, timestamp: str) -> bool:
        """Check if cache entry is expired"""
        try:
            entry_time = datetime.fromisoformat(timestamp)
            return datetime.now() - entry_time > self.cache_ttl
        except:
            return True
    
    def find_similar_cached_query(
        self,
        query: str,
        selected_document: Optional[str] = None,
        similarity_threshold: float = 0.95
    ) -> Optional[Tuple[str, Dict]]:
        """Find semantically similar cached query"""
        self._clean_expired()
        
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        best_match = None
        best_score = 0.0
        
        for entry in self.cache_index['entries']:
            if entry['selected_document'] != selected_document:
                continue
            
            cached_embedding = np.array(entry['query_embedding'])
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = entry
        
        if best_match:
            cache_file = self.cache_dir / f"{best_match['cache_key']}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"💾 Cache HIT! Similarity: {best_score:.3f}")
                return best_match['cache_key'], cached_data
        
        print("❌ Cache MISS")
        return None
    
    def cache_result(
        self,
        query: str,
        response: str,
        chunks: List[Dict],
        debug_info: Dict,
        selected_document: Optional[str] = None
    ) -> str:
        """Cache query result"""
        cache_key = self._generate_cache_key(query, selected_document)
        
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        entry = {
            'cache_key': cache_key,
            'query': query,
            'query_embedding': query_embedding,
            'selected_document': selected_document,
            'timestamp': datetime.now().isoformat(),
            'hit_count': 0
        }
        
        self.cache_index['entries'] = [
            e for e in self.cache_index['entries']
            if e['cache_key'] != cache_key
        ]
        
        self.cache_index['entries'].append(entry)
        self._save_index()
        
        cache_data = {
            'query': query,
            'response': response,
            'chunks': chunks,
            'debug_info': debug_info,
            'selected_document': selected_document,
            'cached_at': datetime.now().isoformat()
        }
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        return cache_key
    
    def _clean_expired(self) -> None:
        """Remove expired cache entries"""
        valid_entries = []
        
        for entry in self.cache_index['entries']:
            if not self._is_expired(entry['timestamp']):
                valid_entries.append(entry)
            else:
                cache_file = self.cache_dir / f"{entry['cache_key']}.json"
                if cache_file.exists():
                    cache_file.unlink()
        
        if len(valid_entries) != len(self.cache_index['entries']):
            self.cache_index['entries'] = valid_entries
            self._save_index()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        self._clean_expired()
        
        total_entries = len(self.cache_index['entries'])
        total_hits = sum(e.get('hit_count', 0) for e in self.cache_index['entries'])
        
        cache_size_mb = sum(
            (self.cache_dir / f"{e['cache_key']}.json").stat().st_size
            for e in self.cache_index['entries']
            if (self.cache_dir / f"{e['cache_key']}.json").exists()
        ) / (1024 * 1024)
        
        return {
            'total_entries': total_entries,
            'total_hits': total_hits,
            'cache_size_mb': cache_size_mb,
            'hit_rate': total_hits / max(total_entries, 1)
        }
    
    def clear_cache(self) -> None:
        """Clear all cache entries"""
        for entry in self.cache_index['entries']:
            cache_file = self.cache_dir / f"{entry['cache_key']}.json"
            if cache_file.exists():
                cache_file.unlink()
        
        self.cache_index = {'entries': []}
        self._save_index()
        print("🗑️ Cache cleared")
