import pkg_resources

named_objects = {}
for ep in pkg_resources.iter_entry_points(group='example'):
   named_objects.update({ep.name: ep.load()})

named_objects['hello']()