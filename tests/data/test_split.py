from ancora.data import split


def test_find_leaks():
    rows = [('song', 'Tester', 'Test Song'), ('other', 'Tester', 'Test Song (Live)')]
    assert split.find_leaks({'train': ['song'], 'val': [], 'test': ['other']}, rows)
    assert not split.find_leaks({'train': ['song', 'other'], 'val': [], 'test': []}, rows)
