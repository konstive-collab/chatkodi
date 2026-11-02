from core.agent.fitness_assessment import read_profiles, select, assess, composition


def source():
    header = ['№ п/п','Военнослужащие','Номер категории','Минимум баллов в одном упражнении'] + ['']*16
    counts = ['']*20
    for i,value in [(4,'В двух упражнениях'),(7,'В трех упражнениях'),(10,'В четырех упражнениях'),(13,'В пяти упражнениях'),
                    (16,'Высший квалификационный уровень'),(17,'Первый квалификационный уровень'),(18,'Второй квалификационный уровень'),(19,'Третий квалификационный уровень')]: counts[i]=value
    row = ['','3 и последующих курсов','–','38'] + ['–']*9 + ['360','320','260','400','380','370','360']
    rows=[header,counts,['2','Курсанты:']+['']*18,row]
    text='не набран минимум баллов в одном физическом упражнении\n--- Страница 139 ---\nПриложение № 6\n'
    text+='| '+' | '.join(rows[0])+' |\n|'+'---|'*20+'\n'
    text+='\n'.join('| '+' | '.join(r)+' |' for r in rows[1:])
    return [('НФП.pdf',text)]


def test_minimum_and_five_exercise_requirement_are_read_from_the_supplied_table():
    profiles=read_profiles(source())
    assert len(profiles)==1
    p=profiles[0]
    assert p.minimum==38 and p.count==5 and p.source=='НФП.pdf, стр. 139'
    assert p.grades==[('5',360),('4',320),('3',260)]
    assert assess(p,[40,55])[:2]==('Н/Д','Н/Д')
    assert assess(p,[86,34])[:2]==(2,'нет')
    assert assess(p,[])[:2]==('Н/Д','Н/Д')


def test_complete_positive_grade_requires_valid_exercise_composition():
    p=read_profiles(source())[0]
    valid={'1':('ОФУ','сила'),'17':('ОФУ','быстрота'),'30':('ОФУ','ловкость'),'43':('СФУ',''),'61':('СФУ','')}
    ids=['1','17','30','43','61']
    assert assess(p,[80]*5,ids,valid)[:2]==(5,'ВК')
    assert assess(p,[80]*5,['1']*5,valid)[0]=='Н/Д'
    assert assess(p,[80]*5,ids,{})[0]=='Н/Д'


def test_cadet_course_is_not_confused_with_contract_category():
    profiles=read_profiles(source())
    assert select(profiles,['Курс обучения','Пол'],['3 курс','м']) is not None
    assert select(profiles,['Курс обучения','Пол'],['3','м']) is None
    assert select(profiles,['Курс обучения','Пол'],['3 курс','ж']) is None
